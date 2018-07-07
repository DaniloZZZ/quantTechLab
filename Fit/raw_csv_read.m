
apply = @(f,x)cell2mat(arrayfun(f,x,'UniformOutput',false))

arr1 = csvread('data_05-07-2018_20-20_port0.csv');
arr2 = csvread('data_05-07-2018_20-30_port1.csv');
arr3 = csvread('data_05-07-2018_20-20_port2.csv');
data = [arr1(:, 1),arr2(:, 1),arr3(:, 1)];
time = arr(:, 2);

