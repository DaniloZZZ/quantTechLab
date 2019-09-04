T = 0.6;
t=sqrt(T);
r=sqrt(1-T);

mod = [33;39;39];
mod = mod/norm(mod);

st_params=num2cell([ 0.125,0,pi,0,pi,pi,pi,0,0,0,0,0]);
[a1,al1,al2,al3,al4,ala,alb,th1,th2,th3,th4,thb]=deal(st_params{:});


a01 = @(al,thet)...
[
	t r*exp(1i*al) 0;
	-r*exp(-1i*al) t 0;
	0 0 exp(-1i*thet)
];


a12 = @(al,thet)...
[
	exp(-1i*thet)  0 0;
	0 t r*exp(-1i*al ) ;
	0 -r*exp(-1i*al) t
];

a02 = @(al,thet)...
[
	t 0 -r*exp(-1i*al) ;
	0 exp(-1i*thet)  0;
	-r*exp(-1i*al) 0 t
];

refl0=@(k) diag([sqrt(1-k),1,1]);
refl1=@(k) diag([1,sqrt(1-k),1]);
refl2=@(k) diag([1,1,sqrt(1-k)]);

ph = @(x)diag(exp(1i*x));
ph2 = @(ph) diag([1,exp(1i*ph),exp(1i*2*ph)]);

psi = 1.33;

device= @(x1,x2,x3,x4,phi)...
[0,1,1]*a12(al4,th4)...
*ph([0, atan(1.41)-2*psi+al4-th3-th2-th1-al1+al2+alb+x4,0])...
*refl1(a1)...
*a01(al3,th3)...
*ph([-2*atan(1.41)+al3-al2+pi+x3,0,0])...
*refl0(a1)...
*ph([psi,psi,0])...
*a01(al2,th2)...
*ph([0,-al2+pi/2.-th1-alb+x2,0])...
*refl1(a1)...
*ph([ala+thb+psi,psi,0])...
*a12(al1,th1)...
*ph([0,0,-al1-psi+ala+thb+alb+x1])...
*refl2(a1)...
*ph([0,ala+thb+alb+x1,psi])...
*ph2(phi)...
*mod...

fourier = @(x) device(0,0,0,0,x);

x = 0:6/40:6;
B = arrayfun(fourier,x,'UniformOutput',false);
E = cell2mat(B);
y = abs(power(E,2));

abssq=@(x)abs(power(x,2));
funct= @(x1,x2,x3,x4,x)cell2mat(arrayfun(@(x)abssq(device(x1,x2,x3,x4,x)),x,'UniformOutput',false));
exp_data = y + (rand(size(y))-0.5)/4;
