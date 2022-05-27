@cuda.jit
dev calc_acc_cuda(acc, r, mass, z, epsilon):
    i = cuda.grid(1)
    if i < r.size:
        acc[i] = r[i] * mass[i] / (z[i] + epsilon **2) ** (1.5)
	

def calc_acc(acc, pos, mass):

	epsilon = 1.1*np.power(len(pos), -0.48)
    
	for i in range(len(pos)):
        r = pos[:, :] - pos[i, :]
        
		z = r[:, 0] ** 2 + r[:, 1] ** 2
		
		calc_acc_cuda(acc_x, r[:,0], mass, z, epsilon)
		calc_acc_cuda(acc_y, r[:,1], mass, z, epsilon)
		
		acc[i,0] = np.sum(acc_x)
		acc[i,1] = np.sum(acc_y)
		