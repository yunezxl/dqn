import numpy as np
import os
import pickle
import scipy.stats as stats
from scipy.stats import norm, expon
import pandas as pd
import random

def GenerationData(DisType_Demand, DisParams_Demand, TrainSampleSize):
    warmup = 0
    TestSampleSize = 60 + warmup
    rho = 0.5

    if DisType_Demand == "RealData":
        TrainSamples = []
        TestSamples = []

        # df = pd.read_excel('D:\\桌面\\Fish\\POM------一审\\JD.xlsx')
        # for i in range(2):
        #     if i == 1:
        #         column_name = '2ddb64e05a'
        #         selected_columns = df[column_name]
        #         split_index = int(len(selected_columns) * 20 / 31)
        #         item_train_data = df.iloc[:split_index]
        #         item_test_data = df.iloc[split_index:]
        #         item_train_data = item_train_data['2ddb64e05a'].tolist()
        #         item_test_data = item_test_data['2ddb64e05a'].tolist()
        #     if i == 0:
        #         column_name = 'cdee05b50c'
        #         selected_columns = df[column_name]
        #         split_index = int(len(selected_columns) * 20 / 31)
        #         item_train_data = df.iloc[:split_index]
        #         item_test_data = df.iloc[split_index:]
        #         item_train_data = item_train_data['cdee05b50c'].tolist()
        #         item_test_data = item_test_data['cdee05b50c'].tolist()
        #     TrainSamples.append(item_train_data)
        #     TestSamples.append(item_test_data)

        df = pd.read_csv(open('D:\\桌面\\Fish\\POM------一审\\Kaggle 2018 数据.csv'))
        for i in range(2):
            if i == 0:
                item_data = df[df['item'] == 26]
                train_data = []
                test_data = []
                for index, row in item_data.iterrows():
                    if len(train_data) < int(len(item_data) * 0.75):
                        train_data.append(row)
                    else:
                        test_data.append(row)
                train_data = pd.DataFrame(train_data)
                test_data = pd.DataFrame(test_data)
                train_data = train_data['sales'].tolist()
                test_data = test_data['sales'].tolist()
            if i == 1:
                item_data = df[df['item'] == 40]
                train_data = []
                test_data = []
                for index, row in item_data.iterrows():
                    if len(train_data) < int(len(item_data) * 0.75):
                        train_data.append(row)
                    else:
                        test_data.append(row)
                train_data = pd.DataFrame(train_data)
                test_data = pd.DataFrame(test_data)
                train_data = train_data['sales'].tolist()
                test_data = test_data['sales'].tolist()
            TrainSamples.append(train_data)
            TestSamples.append(test_data)


    if DisType_Demand == "Normal":
        Samples = np.random.multivariate_normal(DisParams_Demand[0],
                                                DisParams_Demand[1],
                                                TrainSampleSize + TestSampleSize)
        Samples = np.round(Samples)
        Samples = np.clip(Samples, 0, 300)
        TrainSamples = Samples[:TrainSampleSize]
        TrainSamples = np.array(TrainSamples).reshape(-1, 2).transpose()
        TestSamples = Samples[TrainSampleSize:]
        TestSamples = np.array(TestSamples).reshape(-1, 2).transpose()

    elif DisType_Demand == "Exponential":
        TrainSamples = [[None for _ in range(TrainSampleSize)] for _ in range(2)]
        for i in range(TrainSampleSize):
            correlation_matrix = np.array([[1, rho], [rho, 1]])
            L = np.linalg.cholesky(correlation_matrix)
            standard_uniform_samples = np.random.uniform(size=(1, 2))
            standard_normal_samples = norm.ppf(standard_uniform_samples)
            correlated_normal_samples = np.dot(standard_normal_samples, L)
            exponential_samples_1 = expon.ppf(norm.cdf(correlated_normal_samples[:, 0]), scale=DisParams_Demand[0])
            exponential_samples_2 = expon.ppf(norm.cdf(correlated_normal_samples[:, 1]), scale=DisParams_Demand[1])
            TrainSamples[0][i] = int(np.clip(exponential_samples_1, 0, 300))
            TrainSamples[1][i] = int(np.clip(exponential_samples_2, 0, 300))

        TestSamples = [[None for _ in range(TestSampleSize)] for _ in range(2)]
        for i in range(TestSampleSize):
            correlation_matrix = np.array([[1, rho], [rho, 1]])
            L = np.linalg.cholesky(correlation_matrix)
            standard_uniform_samples = np.random.uniform(size=(1, 2))
            standard_normal_samples = norm.ppf(standard_uniform_samples)
            correlated_normal_samples = np.dot(standard_normal_samples, L)
            exponential_samples_1 = expon.ppf(norm.cdf(correlated_normal_samples[:, 0]),
                                              scale=DisParams_Demand[0])
            exponential_samples_2 = expon.ppf(norm.cdf(correlated_normal_samples[:, 1]),
                                              scale=DisParams_Demand[1])
            TestSamples[0][i] = int(np.clip(exponential_samples_1, 0, 220))
            TestSamples[1][i] = int(np.clip(exponential_samples_2, 0, 220))



    elif DisType_Demand == "Mix":
        TrainSamples = [[None for _ in range(TrainSampleSize)] for _ in range(2)]
        for i in range(TrainSampleSize):
            distribution = np.random.choice([0,1,2])
            if distribution == 0:
                temp_sample = np.random.multivariate_normal(DisParams_Demand[0][0], DisParams_Demand[0][1], 1)
                temp_sample = np.clip(temp_sample, 0, 300)
                TrainSamples[0][i] = int(temp_sample[0][0])
                TrainSamples[1][i] = int(temp_sample[0][1])
            elif distribution == 1:
                correlation_matrix = np.array([[1, rho], [rho, 1]])
                L = np.linalg.cholesky(correlation_matrix)
                standard_uniform_samples = np.random.uniform(size=(1, 2))
                standard_normal_samples = norm.ppf(standard_uniform_samples)
                correlated_normal_samples = np.dot(standard_normal_samples, L)
                exponential_samples_1 = expon.ppf(norm.cdf(correlated_normal_samples[:, 0]), scale=DisParams_Demand[1][0])
                exponential_samples_2 = expon.ppf(norm.cdf(correlated_normal_samples[:, 1]), scale=DisParams_Demand[1][1])
                TrainSamples[0][i] = int(np.clip(exponential_samples_1, 0, 300))
                TrainSamples[1][i] = int(np.clip(exponential_samples_2, 0, 300))
            else:
                correlation_matrix = np.array([[1, rho], [rho, 1]])
                L = np.linalg.cholesky(correlation_matrix)
                standard_uniform_samples = np.random.uniform(size=(1, 2))
                correlated_samples = np.dot(standard_uniform_samples, L)
                scaled_samples = np.array([
                    correlated_samples[:, 0] * (200 / np.sqrt(3)) + 100,
                    correlated_samples[:, 1] * (280 / np.sqrt(3)) + 140
                ])
                TrainSamples[0][i] = int(scaled_samples[0])
                TrainSamples[1][i] = int(scaled_samples[1])

        TestSamples = [[None for _ in range(TestSampleSize)] for _ in range(2)]
        for i in range(TestSampleSize):
            distribution = np.random.choice([0, 1, 2])
            if distribution == 0:
                temp_sample = np.random.multivariate_normal(DisParams_Demand[0][0],
                                                            DisParams_Demand[0][1],
                                                            1)
                temp_sample = np.clip(temp_sample, 0, 300)
                TestSamples[0][i] = int(temp_sample[0][0])
                TestSamples[1][i] = int(temp_sample[0][1])
            elif distribution == 1:
                correlation_matrix = np.array([[1, rho], [rho, 1]])
                L = np.linalg.cholesky(correlation_matrix)
                standard_uniform_samples = np.random.uniform(size=(1, 2))
                standard_normal_samples = norm.ppf(standard_uniform_samples)
                correlated_normal_samples = np.dot(standard_normal_samples, L)
                exponential_samples_1 = expon.ppf(norm.cdf(correlated_normal_samples[:, 0]),
                                                  scale=DisParams_Demand[1][0])
                exponential_samples_2 = expon.ppf(norm.cdf(correlated_normal_samples[:, 1]),
                                                  scale=DisParams_Demand[1][1])
                TestSamples[0][i] = int(np.clip(exponential_samples_1, 0, 300))
                TestSamples[1][i] = int(np.clip(exponential_samples_2, 0, 300))
            else:
                correlation_matrix = np.array([[1, rho], [rho, 1]])
                L = np.linalg.cholesky(correlation_matrix)
                standard_uniform_samples = np.random.uniform(size=(1, 2))
                correlated_samples = np.dot(standard_uniform_samples, L)
                scaled_samples = np.array([
                    correlated_samples[:, 0] * (200 / np.sqrt(3)) + 100,
                    correlated_samples[:, 1] * (280 / np.sqrt(3)) + 140
                ])
                TestSamples[0][i] = int(scaled_samples[0])
                TestSamples[1][i] = int(scaled_samples[1])

    elif DisType_Demand == "Cyclic":
        TrainSamples = []
        for _ in range(int(TrainSampleSize / 5)):
            temp_sample1 = np.random.multivariate_normal(DisParams_Demand[0][0],
                                                         DisParams_Demand[0][1],
                                                         4)
            temp_sample1 = np.round(temp_sample1)
            temp_sample1[:, 0] = np.clip(temp_sample1[:, 0], 0, 300)
            temp_sample1[:, 1] = np.clip(temp_sample1[:, 1], 0, 300)
            TrainSamples.extend(temp_sample1.tolist())

            correlation_matrix = np.array([[1, rho], [rho, 1]])
            L = np.linalg.cholesky(correlation_matrix)
            standard_uniform_samples = np.random.uniform(size=(1, 2))
            standard_normal_samples = norm.ppf(standard_uniform_samples)
            correlated_normal_samples = np.dot(standard_normal_samples, L)
            temp_sample2 = expon.ppf(norm.cdf(correlated_normal_samples), scale=DisParams_Demand[1])
            temp_sample2 = np.array(temp_sample2)
            temp_sample2[:, 0] = np.clip(temp_sample2[:, 0], 0, 300)
            temp_sample2[:, 1] = np.clip(temp_sample2[:, 1], 0, 300)
            for sublist in temp_sample2:
                TrainSamples.append(sublist.tolist())
        TrainSamples = np.array(TrainSamples).reshape(-1, 2).transpose()


        TestSamples = []
        for _ in range(int(TestSampleSize / 5)):
            temp_sample1 = np.random.multivariate_normal(DisParams_Demand[0][0],
                                                         DisParams_Demand[0][1],
                                                         4)
            temp_sample1 = np.round(temp_sample1)
            temp_sample1[:, 0] = np.clip(temp_sample1[:, 0], 0, 220)
            temp_sample1[:, 1] = np.clip(temp_sample1[:, 1], 0, 220)
            TestSamples.extend(temp_sample1.tolist())
            correlation_matrix = np.array([[1, rho], [rho, 1]])
            L = np.linalg.cholesky(correlation_matrix)
            standard_uniform_samples = np.random.uniform(size=(1, 2))
            standard_normal_samples = norm.ppf(standard_uniform_samples)
            correlated_normal_samples = np.dot(standard_normal_samples, L)
            temp_sample2 = expon.ppf(norm.cdf(correlated_normal_samples), scale=DisParams_Demand[1])
            temp_sample2 = np.array(temp_sample2)
            temp_sample2[:, 0] = np.clip(temp_sample2[:, 0], 0, 300)
            temp_sample2[:, 1] = np.clip(temp_sample2[:, 1], 0, 300)
            for sublist in temp_sample2:
                TestSamples.append(sublist.tolist())
        TestSamples = np.array(TestSamples).reshape(-1, 2).transpose()

    else:
        pass

    return TrainSamples, TestSamples

def SaveData(Data, filename, path):
    file_path = os.path.join(path, filename)
    with open(file_path, 'wb') as file:
        pickle.dump(Data, file)

