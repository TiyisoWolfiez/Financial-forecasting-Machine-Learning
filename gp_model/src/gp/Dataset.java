package gp;

import java.io.*;
import java.util.*;

public class Dataset {
    private double[][] data;
    private int[] labels;
    private int numFeatures;

    public Dataset(String filename) throws IOException {
        load(filename);
    }

    private void load(String filename) throws IOException {
        List<double[]> dataList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();

        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        line = reader.readLine();
        while ((line = reader.readLine()) != null) {
            if (line.trim().isEmpty()) continue;

            String[] parts = line.split(",");
            int len = parts.length;
            double[] features = new double[len - 1];

            for (int i = 0; i < len - 1; i++) {
                features[i] = Double.parseDouble(parts[i].trim());
            }

            int label = Integer.parseInt(parts[len - 1].trim());
            dataList.add(features);
            labelList.add(label);
        }
        reader.close();

        data = new double[dataList.size()][];
        labels = new int[labelList.size()];
        for (int i = 0; i < dataList.size(); i++) {
            data[i] = dataList.get(i);
            labels[i] = labelList.get(i);
        }
        numFeatures = data[0].length;
    }

    public double[][] getData() {
        return data;
    }

    public int[] getLabels() {
        return labels;
    }

    public int getFeatureCount() {
        return numFeatures;
    }

    public int size() {
        return data.length;
    }
}