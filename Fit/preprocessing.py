"""
Created by Danil Lykov ( @danlkv ) at 07/02/2019

File for preprocessing the input data

## Data structure:
every line is a measurment of laser beam intensity
every line has a format

`value,time`

"""

import numpy as np

def data_read(filenames):
	"""
	Reads the data from filenames
	Returns
	--------------
	[(vals,times),(vals,times), ] : resulting unscaled data
	"""
	raw_data = [np.genfromtxt(fname, delimiter=',') for fname in filenames]
	data = [ (filedata[:,0],filedata[:,1]) for filedata in raw_data ]
	#times = [np.array(i.T[1]) for i in exp_points]
	#exp_points = [np.array(i.T[0]) for i in exp_points]
	return data

def data_preprocess(data,params):
	"""
	loads data, scales it and crops.
	Returns
	--------------
	(x_transformed,y_transformed)
	"""
	def transformer(data, scaling, shifting):
		assert len(data)==len(scaling)==len(shifting), "transform params different dimensions"
		def transform(x, scale, shift):
			y = (x + shift)*scale
			return y
		return [transform(*i) for i in zip(data,scaling,shifting)]

	def clipper(data,clip_range):
		"""
		Clips by range with respect to second (x) array of data
		"""
		if clip_range:
			mask = np.array([clip_range[0]<dp<clip_range[1] for dp in data[1]])
			return (data[0][mask], data[1][mask])
		else:
			return data

	min_, ampl, per = params['min_amp_per']
	start_exp, end_exp = params['start_end_index']

	transformed = []
	for channel,scale in zip(data,params['scaling_channels']):
		scaling = (1/scale, 2*np.pi/per)
		shifting = ( -min_, -data[0][1][start_exp])
		ranging = (channel[1][start_exp], channel[1][end_exp])
		clipped = clipper(channel, ranging)
		scaled_shifted = transformer(clipped,scaling,shifting)
		transformed.append(scaled_shifted)
	return transformed

"""
These are data to read experiments and scale the values to good view.
"""
handmade_exp = {
		'20_07_2018':{
			'filenames': [
				'./20_07/data_20-07-2018_22-41_port0.csv',
				'./20_07/data_20-07-2018_22-41_port1.csv',
				'./20_07/data_20-07-2018_22-44_port2.csv'
				],
			'exp_params' : {
				'scaling_channels':[0.99,1.0,1.0],
				'min_amp_per':[10,80-10,75],
				'start_end_index':[-160,-10],
				}
			},
	'20_11_2018_16-02':{
		'filenames': [
			'./data_11/data_20-11-2018_16-02_port0.csv',
			'./data_11/data_20-11-2018_16-02_port1.csv',
			'./data_11/data_20-11-2018_16-02_MPD_port2.csv'
			],
		'exp_params' : {
			'scaling_channels':[3,2,500],
			'min_amp_per':[0,10,300/4],
			'start_end_index':[-160,-10],
			}
		},
	}
def read_data_params(exp):
    fnames = exp['filenames']
    params = exp['exp_params']

    raw_data = data_read(fnames)
    data = data_preprocess(raw_data,params)
    print(">> Read %i channels with lengths %i"%(len(data),len(data[0])))
    return data

def read_data_name(data_name='20_11_2018_16-02'):
    exp = handmade_exp[data_name]
    return read_data_params(exp)

def check_time_shift(data):
    shifts = [ max(data[0][1]-data[1][1]),
                max(data[1][1]-data[2][1]),
                max(data[0][1]-data[2][1]),
             ]
    print("Max delta of x-points in domain 0-1:", shifts[0])
    print("Max delta of x-points in domain 1-2:", shifts[1])
    print("Max delta of x-points in domain 0-2:", shifts[2])
    dx = data[0][1][1:]-data[0][1][:-1]
    mean0 = np.mean(dx)
    print("x step on domain 0:", np.mean(dx),'variance of step:', np.std(dx))
    dx = data[1][1][1:]-data[1][1][:-1]
    mean1 = np.mean(dx)
    print("x step on domain 1:", np.mean(dx),'variance of step:', np.std(dx))
    dx = data[2][1][1:]-data[2][1][:-1]
    mean2 = np.mean(dx)
    print("x step on domain 2:", np.mean(dx),'variance of step:', np.std(dx))
    max_step = max([mean0,mean1,mean2])
    max_shift = max(np.abs(shifts))
    print("\nmax channel domain shift: %f\nmax domain step: %f"%(max_shift,max_step))
    if max_step*0.1 < max_shift:
        print("WARNING max shift is %f of domain step"%(max_shift/max_step))

def main():
    data = read_data()
    print(data)

if __name__=='__main__':
	main()
