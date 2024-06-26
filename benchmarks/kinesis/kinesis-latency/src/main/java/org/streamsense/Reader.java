package org.streamsense;

import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.cloudwatch.CloudWatchAsyncClient;
import software.amazon.awssdk.services.dynamodb.DynamoDbAsyncClient;
import software.amazon.awssdk.services.kinesis.KinesisAsyncClient;
import software.amazon.kinesis.common.ConfigsBuilder;
import software.amazon.kinesis.coordinator.Scheduler;
import software.amazon.kinesis.lifecycle.events.*;
import software.amazon.kinesis.metrics.MetricsLevel;
import software.amazon.kinesis.processor.RecordProcessorCheckpointer;
import software.amazon.kinesis.processor.ShardRecordProcessor;
import software.amazon.kinesis.processor.ShardRecordProcessorFactory;
import software.amazon.kinesis.retrieval.KinesisClientRecord;

import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.concurrent.*;

public class Reader {

    String streamName;
    String region;
    KinesisAsyncClient kinesisClient;
    DynamoDbAsyncClient dynamoClient;
    CloudWatchAsyncClient cloudWatchClient;

    int numRecords;
    CountDownLatch barrier;

    public Reader(String region, String streamName, int numRecords, CountDownLatch barrier) {
        this.region = region;
        this.streamName = streamName;
        Region r = Region.of(this.region);
        this.kinesisClient = KinesisAsyncClient.builder().region(r).build();
        dynamoClient = DynamoDbAsyncClient.builder().region(r).build();
        cloudWatchClient = CloudWatchAsyncClient.builder().region(r).build();
        this.barrier = barrier;

        this.numRecords = numRecords;
    }

    public ArrayList<Long> read() {
        CountDownLatch latch = new CountDownLatch(numRecords);
        String applicationName = "Reader-" + System.currentTimeMillis();
        String workerID = "reader-" + String.format("%03d", 0);

        ArrayList<Long> timestamps = new ArrayList<>();


        ConfigsBuilder configsBuilder = new ConfigsBuilder(streamName,
                applicationName,
                kinesisClient,
                dynamoClient,
                cloudWatchClient,
                workerID,
                new RecordProcessorFactory(timestamps, latch));


        Scheduler scheduler = new Scheduler(
                configsBuilder.checkpointConfig(),
                configsBuilder.coordinatorConfig(),
                configsBuilder.leaseManagementConfig(),
                configsBuilder.lifecycleConfig(),
                configsBuilder.metricsConfig().metricsLevel(MetricsLevel.NONE),
                configsBuilder.processorConfig(),
                configsBuilder.retrievalConfig()
        );

        Thread schedulerThread = new Thread(scheduler);
        schedulerThread.setDaemon(true);
        schedulerThread.start();

        try {
            latch.await();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        Future<Boolean> gracefulShutdownFuture = scheduler.startGracefulShutdown();
        System.out.println("Waiting up to 20 seconds for shutdown to complete.");
        try {
            gracefulShutdownFuture.get(20, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            System.out.println("Interrupted while waiting for graceful shutdown. Continuing.");
        } catch (ExecutionException e) {
            System.out.println("Exception while executing graceful shutdown.");
        } catch (TimeoutException e) {
            System.out.println("Timeout while waiting for shutdown.  Scheduler may not have exited.");
        }
        System.out.println("Completed, shutting down now reader.");

        return timestamps;
    }

    private class RecordProcessorFactory implements ShardRecordProcessorFactory {
        ArrayList<Long> timestamps;
        CountDownLatch latch;

        public RecordProcessorFactory(ArrayList<Long> timestamps, CountDownLatch latch){
            this.timestamps = timestamps;
            this.latch = latch;
        }

        @Override
        public ShardRecordProcessor shardRecordProcessor() {
            return new ImplShardRecordProcessor(timestamps, latch);
        }
    }

    private class ImplShardRecordProcessor implements ShardRecordProcessor {

        ArrayList<Long> timestamps;
        CountDownLatch latch;
        MessageDigest digest;

        public ImplShardRecordProcessor(ArrayList<Long> timestamps, CountDownLatch latch){
            try {
                digest = MessageDigest.getInstance("SHA-256");
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            this.timestamps = timestamps;
            this.latch = latch;
        }

        @Override
        public void initialize(InitializationInput initializationInput) {
            System.out.println("Initializing record processor for shard: " + initializationInput.shardId());
            barrier.countDown();
        }

        @Override
        public void processRecords(ProcessRecordsInput processRecordsInput) {
            for (KinesisClientRecord record : processRecordsInput.records()) {
                ByteBuffer buffer = ByteBuffer.allocate(record.data().capacity());
                record.data().rewind(); // if needed
                buffer.put(record.data());
                buffer.flip(); // if needed
                processRecord(buffer);
            }
        }

        @Override
        public void leaseLost(LeaseLostInput leaseLostInput) {
            System.out.println("Lease lost.");
        }

        @Override
        public void shardEnded(ShardEndedInput shardEndedInput) {
            System.out.println("Shard ended.");
            checkpoint(shardEndedInput.checkpointer());
        }

        @Override
        public void shutdownRequested(ShutdownRequestedInput shutdownRequestedInput) {
            System.out.println("Shutdown requested.");
            checkpoint(shutdownRequestedInput.checkpointer());
        }

        public void processRecord(ByteBuffer record){
            long now = System.currentTimeMillis();
            long value = record.getLong();
            timestamps.add(now - value);
//            System.out.println("Record: " + value + " - " + (now - value) + " ms");
            latch.countDown();
        }
        public void checkpoint(RecordProcessorCheckpointer checkpointer){
            try {
                checkpointer.checkpoint();
            } catch (Exception e) {
                System.err.println("Error checkpointing record: ");
                e.printStackTrace();
            }
        }
    }

}
