package org.streamsense;


import org.apache.commons.cli.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.*;


public class LatencyBenchmark {


    public static void main(String... args) {
        LatencyBenchmark benchmark = new LatencyBenchmark();
        benchmark.parseArgs(args);
        benchmark.eventSize = 100;
        benchmark.run();
        benchmark.run();
        benchmark.run();

        benchmark.eventSize = 1024;
        benchmark.run();
        benchmark.run();
        benchmark.run();

        benchmark.eventSize = 10240;
        benchmark.run();
        benchmark.run();
        benchmark.run();
    }

    private String region;

    private int fps;
    private int eventSize;
    private int duration;
    private String stream;
    private int sleepSeconds;
    private String resultDir;


    public void parseArgs(String [] args){
        Options options = new Options();
        options.addOption("duration", true, "Number of seconds streaming events.");
        options.addOption("fps", true, "Number of events per second");
        options.addOption("stream", true, "Stream name");
        options.addOption("region", true, "AWS region");
        options.addOption("sleepSeconds", true, "Number of seconds to sleep before writing events. Default: 40");
        options.addOption("resultDir", true, "Directory to save results. Default: /home/ubuntu/");
        options.addOption("help", false, "Print help message. Usage: java -jar <jar> -duration " +
                "<duration> -fps <fps> -stream <stream> -region <region> -numEvents <numEvents> -sleepSeconds <sleepSeconds>");


        // Parse the arguments
        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = null;
        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            throw new RuntimeException(e);
        }
        duration = Integer.parseInt(cmd.getOptionValue("duration", "120"));
        fps = Integer.parseInt(cmd.getOptionValue("fps", "25"));
        stream = cmd.getOptionValue("stream", "test-stream");
        region = cmd.getOptionValue("region", "us-east-1");
        eventSize = Integer.parseInt(cmd.getOptionValue("eventSize", "10240"));
        sleepSeconds = Integer.parseInt(cmd.getOptionValue("sleepSeconds", "10"));
        resultDir = cmd.getOptionValue("resultDir", "/home/ubuntu/");
    }
    private void run() {
        ExecutorService executor = Executors.newFixedThreadPool(2);
        int numEvents = fps * duration;
        CountDownLatch barrier = new CountDownLatch(1);
        Writer writer = new Writer(region, stream, numEvents, fps, eventSize, sleepSeconds, barrier);
        Reader reader = new Reader(region, stream, numEvents, barrier);
        Future writeFuture = executor.submit(writer::write);
        Future< ArrayList<Long> > readFuture = executor.submit(reader::read);
        ArrayList<Long> latencies;

        try {
            latencies = readFuture.get();
            writer.stop();
            writeFuture.get();
            long totalLatency = 0;
            for (long latency : latencies) {
                totalLatency += latency;
            }
            double avgLatency = (double) totalLatency / numEvents;
            System.out.println("Average latency: " + avgLatency + " ms");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        // Save latencies to a CSV file
        try {
            String filename = "kinesis-latencies-recordSize-" + eventSize + "-fps-" + fps + "-" + System.currentTimeMillis() +".csv";
            FileWriter fileWriter = new FileWriter(resultDir + filename);
            for (long latency : latencies) {
                fileWriter.write(latency + "\n");
            }
            fileWriter.close();
            System.out.println("Latencies saved to " + filename);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        executor.shutdown();
    }

    @Override
    public boolean equals(Object obj) {
        return super.equals(obj);
    }
}