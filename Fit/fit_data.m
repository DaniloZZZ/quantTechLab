raw_csv_read

f_func_v

fitted = @(x) device(8.805,-5.11,9.458,6.987,x);
x = 0:12/40:12;
B = arrayfun(fitted,x,'UniformOutput',false);
E = cell2mat(B);
y = abs(power(E,2));

from=25;
to =50;
d = data(from:to,2)+data(from:to,3);
x = time(from:to);
start = [ 10.75,-9.439,-1.184,3.5 ,0.033,70,130,-6]
start_2 = [0.05,10,150,6]

tofit = @(x1,x2,x3,x4,w,a,b,ph,x)funct(x1,x2,x3,x4,w*(x-ph))*a+b;
tofit_2 = @(w,a,b,ph,x)funct(8.805,-5.11,9.458,6.987, w*(x-ph))*a+b;
fi = fit(x,d,tofit,...
'StartPoint',start,...
'MaxIter',10000,...
'MaxFunEvals',20000,...
'TolFun',1e-12,...
'TolX',1e-12)


