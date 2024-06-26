package org.streamsense;

import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.kinesis.KinesisAsyncClient;
import software.amazon.awssdk.services.kinesis.model.PutRecordRequest;

import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.*;

public class Writer {

    String streamName;
    String region;
    KinesisAsyncClient kinesisClient;
    Random random = new Random();

    int numRecords;
    int fps;
    int recordSize;
    int sleepSeconds;
    MessageDigest digest;
    CountDownLatch barrier;
    CountDownLatch stopLatch = new CountDownLatch(1);

    public Writer(String region, String streamName, int numRecords, int fps, int recordSize, int sleepSeconds, CountDownLatch barrier) {
        this.region = region;
        this.streamName = streamName;
        Region r = Region.of(region);
        this.kinesisClient = KinesisAsyncClient.builder().region(r).build();
        this.numRecords = numRecords;
        this.fps = fps;
        this.recordSize = recordSize;
        this.sleepSeconds = sleepSeconds;
        this.barrier = barrier;

        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void stop(){
        stopLatch.countDown();
    }

    private void writeRecord() {
        long value = System.currentTimeMillis();
        ByteBuffer buffer = ByteBuffer.allocate(recordSize);
        buffer.putLong(value); // Write the long value into the byte buffer

        // write recordSize - 8 = 92 bytes of random data
        byte[] randomData = new byte[recordSize - 8];
        random.nextBytes(randomData);
        buffer.put(randomData);

        // Generate a random payload
        byte[] payload = buffer.array();
        // Put the record into the stream
        kinesisClient.putRecord(PutRecordRequest.builder()
                .streamName(streamName)
                .partitionKey(String.valueOf(0))
                .data(SdkBytes.fromByteArray(payload))
                .build()).join();
    }

    public void write() {

        System.out.println("Waiting for the reader to be ready...");
        try {
            barrier.await();
            Thread.sleep(sleepSeconds * 1000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }


        System.out.println("Writing " + numRecords + " records to stream " + streamName + " at " + fps + " fps");
        System.out.println("Record size: " + recordSize + " bytes");

        ScheduledExecutorService executor = Executors.newScheduledThreadPool(fps);
        executor.scheduleAtFixedRate(this::writeRecord, 0, 1000 / fps, TimeUnit.MILLISECONDS);

        try {
            stopLatch.await();
            System.out.println("Finished writing records to stream " + streamName);

        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        executor.shutdown();
    }

    public static void main(String[] args) {
        long value = System.currentTimeMillis();
        ByteBuffer buffer = ByteBuffer.allocate(100);
        buffer.putLong(value); // Write the long value into the byte buffer

        // write 100 - 8 = 92 bytes of random data
        byte[] randomData = new byte[92];
        new Random().nextBytes(randomData);
        buffer.put(randomData);

        // Generate a random payload
        byte[] payload = buffer.array();

        // print payload in hex
        for (byte b : payload) {
            System.out.print(String.format("%02X", b));
        }

        ByteBuffer buffer2 = ByteBuffer.wrap(payload);
        long value2 = buffer2.getLong();

        // compare the two values
        System.out.println("\n" + value + " " + value2);
    }

}
