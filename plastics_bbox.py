import os
from pathlib import Path
from argparse import ArgumentParser
import h5py
import pandas as pd
import os
# import random
from numpy import random
import matplotlib.pyplot as plt
import math
import numpy as np
import imageio
from glob import glob

# datadir_path = '/data/LOMUQ'
# resultdir_path = '/data/LOMUQ/jssarna'
datadir_path = "/media/christian/DATA/data/LOMUQ/in"
resultdir_path = "/media/christian/DATA/data/LOMUQ/processed"


class BoundingBox(object):
    def __init__(self, *args, **kwargs):
        self.lat_min = None
        self.lon_min = None
        self.lat_max = None
        self.lon_max = None


def get_bounding_box(longitude_in_degrees, latitude_in_degrees, half_side_in_miles):
    assert half_side_in_miles > 0
    assert longitude_in_degrees >= -180.0 and longitude_in_degrees <= 180.0
    assert latitude_in_degrees >= -90.0 and latitude_in_degrees  <= 90.0

    half_side_in_km = half_side_in_miles * 1.609344
    lon = math.radians(longitude_in_degrees)
    lat = math.radians(latitude_in_degrees)

    radius  = 6371
    # Radius of the parallel at given latitude
    parallel_radius = radius*math.cos(lat)

    lon_min = lon - half_side_in_km/parallel_radius
    lon_max = lon + half_side_in_km/parallel_radius
    lat_min = lat - half_side_in_km/radius
    lat_max = lat + half_side_in_km/radius
    rad2deg = math.degrees

    box = BoundingBox()
    box.lat_min = rad2deg(lat_min)
    box.lon_min = rad2deg(lon_min)
    box.lat_max = rad2deg(lat_max)
    box.lon_max = rad2deg(lon_max)

    return (box)



#initializing    
#Reformat long-lat in x-y format for image

class BoundedParticleCount:
    
    def __init__(self,lat,long,bbox,sample_size,pFile,uFile,vFile,key):
        self.lat = lat
        self.long = long
        self.bbox = bbox
        self.sample_size = sample_size
        self.u_file = uFile
        self.v_file = vFile
        self.p_file = pFile
        self.key = key
        
    
    def boundedBoxRandomParticles(self):

        #extends = (float(px_attrs['min']), float(px_attrs['max']), float(py_attrs['min']), float(py_attrs['max']))
        bound_box = get_bounding_box(self.long,self.lat,self.bbox)
        # inside_boundingbox=[]
        inside_boundingbox_random = []
        randomList=[]
        #Timesteps

        # u_file = h5py.File(self.u_file, "r")
        # timeframe = u_file['uo'].shape[0]

        f = h5py.File(self.p_file, 'r')
        px = np.array(f['p_x'])
        py = np.array(f['p_y'])
        pt = np.array(f['p_t'])
        timeframe = pt.shape[1]
        px0 = px[:, 0].squeeze()
        py0 = py[:, 0].squeeze()
        pt0 = pt[:, 0].squeeze()
        px_valid = ~np.isnan(px0)
        # px0 = px0[px_valid]
        py_valid = ~np.isnan(py0)
        # py0 = py0[py_valid]
        pt_valid = ~np.isnan(pt0)
        # pt0 = pt0[pt_valid]
        x_mask_min = px0 > bound_box.lon_min
        x_mask_max = px0 < bound_box.lon_max
        y_mask_min = py0 > bound_box.lat_min
        y_mask_max = py0 < bound_box.lat_max
        valid_bound_x = np.logical_and(np.logical_and(x_mask_min, x_mask_max), px_valid)
        valid_bound_y = np.logical_and(np.logical_and(y_mask_min, y_mask_max), py_valid)
        total_max = np.logical_and(valid_bound_x, valid_bound_y)
        pts_indices = np.nonzero(total_max)[0]
        # inside_boundingbox = np.array([px[pts_indices], py[pts_indices]]).transpose()
        arr = pts_indices
        
        # arr = []
        # for i in range(len(self.px)):
        #     if bound_box.lat_min <= self.px[i,0] and self.px[i,0] <= bound_box.lat_max and bound_box.lon_min <= self.py[i,0] and self.py[i,0] <= bound_box.lon_max :
        #         # print(px[i,0])
        #         inside_boundingbox.append([self.px[i,0],self.py[i,0]])
        #         arr.append(i)

        # randomList.append(random.sample(arr,round((len(inside_boundingbox)-1)*self.sample_size))) #random sampling
        randomList = random.randint(0, arr.shape[0], round(arr.shape[0] * self.sample_size), dtype=np.uint32)  # random sampling
       
        # for k in range(timeframe):
        #     for i in randomList[0]:
        #         #print(k)
        #         inside_boundingbox_random.append([self.px[i,k],self.py[i,k],k])

        for k in range(timeframe):
            inside_boundingbox_random.append([px[:, k].squeeze()[randomList], py[:, k].squeeze()[randomList]])
        inside_particles = np.array(inside_boundingbox_random).transpose()

        print("# samples: {}".format(inside_particles.shape[0]))

        f.close()

        return inside_particles, randomList
    
    def convertParticlesToParticlesCount(self,inside_boundingbox_random,width,height):

        f = h5py.File(self.p_file, 'r')
        pt = np.array(f['p_t'])
        time = pt.shape[1]

        gres_w = float(width) / 360.0
        gres_h = float(height) / 180.0
        assert gres_h == gres_w
        gres = gres_w
        print("gres: {}".format(gres))
       
        # days_dict = {}
        # for r in inside_boundingbox_random:
        #     try:
        #         days_dict[r[2]].append([r[0],r[1]])
        #     except KeyError:
        #          days_dict[r[2]] = [[r[0],r[1]]]

        # time = len(days_dict)
        
        # particleCount= np.tile(np.zeros([360,720]),(time,1,1)) #Output shape (time, width, height) of zeroes matrices
        # for key in days_dict:
        #     x=[]
        #     y=[]
        #     k = days_dict[key]
        #     for value in k:
        #         lat_bucket = value[0]+90 #Turning -lats to positive i.e range 0,180
        #         lat_bucket = lat_bucket * 100 #18,000 hundredth-degree increments of latitude (i.e. -90.00, -89.99, ... 89.99, 90.00)
        #         lat_bucket = int(round(lat_bucket/(100/2))) #Increment by 0.5 for 360 rows, therefore 50. #50 is dynamic
        #         long_bucket = value[1]+180
        #         long_bucket = long_bucket * 100
        #         long_bucket = int(round(long_bucket/(100/2)))
        #         x.append(lat_bucket) #lat
        #         y.append(long_bucket) #long
        #     for i,j in zip(x,y):
        #         particleCount[key][i,j] += 1
        pcounts = np.zeros((pt.shape[1], height, width), dtype=np.int32)
        for ti in range(pt.shape[1]):
            x_in = inside_boundingbox_random[:, 0, ti].squeeze()
            y_in = inside_boundingbox_random[:, 1, ti].squeeze()
            xpts = np.floor((x_in+(360.0/2.0))*gres).astype(np.int32).flatten()
            ypts = np.floor((y_in+(180.0/2.0))*gres).astype(np.int32).flatten()
            for pi in range(xpts.shape[0]):
                try:
                    pcounts[ti, ypts[pi], xpts[pi]] += 1
                except (IndexError, ) as error_msg:
                    # we don't abort here cause the brownian-motion wiggle of AvectionRK4EulerMarujama always edges on machine precision, which can np.floor(..) make go over-size
                    # print("\nError trying to index point ({}, {}) with indices ({}, {})".format(fX[pi, ti], fY[pi, ti], xpts[pi], ypts[pi]))

                    # print("\nError trying to index point ({}, {}) ...".format(x_in[pi], y_in[pi]))
                    # print("Point index: {}".format(pi))
                    # print("Requested spatial index: ({}, {})".format(xpts[pi], ypts[pi]))

                    pass
            # print("non-zeros: {}".format(np.count_nonzero(pcounts[:, :, ti])))
        # hf.create_dataset('ParticleCount', data=particleCount)

        # print_pcount = np.flip(pcounts,1) #Because matrix indices are different.  Flipping the rows
        # for ti in range(pt.shape[1]):
        #     imageio.imwrite(os.path.join(resultdir_path, self.key, "particleCount_"+str(self.key)+"_"+str(ti)+".ppm"), print_pcount[ti, :, :])
        f.close()
        return pcounts



if __name__ == "__main__":
    
    
    parser = ArgumentParser(description="Program prints randomly sampled particles inside a bounding box")
    parser.add_argument("-lat", "--latitude",  type=float, help="Enter the latitude in arc degrees")
    parser.add_argument("-long", "--longititude", type=float, help="Enter the longititude in arc degrees")
    parser.add_argument("-box", "--boundingbox", type=int, help="Enter the length of a half-side of the bounding box [in miles]")
    parser.add_argument("-size", "--samplesize", type=float, help="Enter the number of samples you want to randomly sample [per particle in box]")
    args = parser.parse_args()

    subfolders = sorted(glob(os.path.join(datadir_path, "*")))

    # pathlist = Path(datadir_path).rglob('*.*')
    # paths = []
    # filesDict = {}
    # for path in pathlist:
    #     path_in_str = str(path)
    #     paths = path_in_str.split('/')
    #     try:
    #         filesDict[paths[-2]].append(paths[-1])
    #     except KeyError:
    #         filesDict[paths[-2]] = [paths[-1]]

    particleCountList=[]
    # for key in filesDict:
    for LOMUQfolder in subfolders:
        try:
            fullpath = LOMUQfolder
            parentfolder = os.path.abspath(Path(LOMUQfolder).parent)
            key = fullpath[len(parentfolder)+1:]
            print("Split folder to '{}' -> {}".format(parentfolder, key))

            # hydrodynamic_U = datadir_path +"/" + key + "/"+"hydrodynamic_U.h5"
            # hydrodynamic_V = datadir_path +"/" + key + "/"+"hydrodynamic_V.h5"
            # particles = datadir_path +"/" + key + "/"+"particles.h5"
            hydrodynamic_U = os.path.join(fullpath, "hydrodynamic_U.h5")
            hydrodynamic_V = os.path.join(fullpath, "hydrodynamic_V.h5")
            particles = os.path.join(fullpath, "particles.h5")
            if not os.path.exists(hydrodynamic_U) or not os.path.exists(hydrodynamic_V) or not os.path.exists(particles):
                continue
            hydrodynamic_U_data = h5py.File(hydrodynamic_U , "r")

            height, width = np.shape(hydrodynamic_U_data['uo'][0])
            # particle_data = h5py.File(particles, "r")
            print(width,height)
            # px = particle_data['p_y'][()]
            # py = particle_data['p_x'][()]

            # bpc = BoundedParticleCount(args.latitude,args.longititude,args.boundingbox,args.samplesize,px,py,key)
            # (self,lat,long,bbox,sample_size,pFile,uFile,vFile,key)
            bpc = BoundedParticleCount(args.latitude,args.longititude,args.boundingbox,args.samplesize, particles, hydrodynamic_U, hydrodynamic_V, key)
            bboundParticles,p_idx = bpc.boundedBoxRandomParticles()
            particleCountList.append(bpc.convertParticlesToParticlesCount(bboundParticles,width,height))
            hydrodynamic_U_data.close()
        except ValueError:
            continue
    
    # r,c = np.shape(particleCountList[0][0])
    t, r, c = np.shape(particleCountList[0])
    print("t, r, c = {}, {}, {}".format(t, r,c))

    #Cropping the "action" area
    # dataDict={}
    # for i in range(len(particleCountList)):
    #     for j in range(len(particleCountList[i])):
    #         for rows in range(0,r-40,40):
    #             for columns in range(0,c-40,40):
    #                 if(particleCountList[i][j][rows:rows+40,columns:columns+40].sum()>0):
    #                     dataDict[(i,j,rows,columns)] = particleCountList[i][j][rows:rows+40,columns:columns+40].sum()
    
    # df = pd.DataFrame(dataDict.keys())
    # df.columns=['Data','day','rows','columns']

    # minRow =  df['rows'].min()
    # maxRow =  df['rows'].max()
    # minCol =  df['columns'].min()
    # maxCol =  df['columns'].max()
    
    # particleCountList = np.asarray(particleCountList)
    # particleCountList = particleCountList[...,minRow:maxRow+40,minCol:maxCol+40]

    minx, maxx = np.iinfo(np.int32).max, np.iinfo(np.int32).min
    miny, maxy = np.iinfo(np.int32).max, np.iinfo(np.int32).min
    for i in range(len(particleCountList)):
        for ti in range(t):
            map = particleCountList[i][ti, :, :].squeeze()
            indices_x = np.nonzero(map)[1]
            indices_y = np.nonzero(map)[0]
            iminx, imaxx = np.min(indices_x), np.max(indices_x)
            print("iminx, imaxx = {}, {}".format(iminx, imaxx))
            iminy, imaxy = np.min(indices_y), np.max(indices_y)
            print("iminy, imaxy = {}, {}".format(iminy, imaxy))
            minx = np.minimum(minx, iminx)
            maxx = np.maximum(maxx, imaxx)
            miny = np.minimum(minx, iminy)
            maxy = np.maximum(maxx, imaxy)
    minRow = miny - (miny % 40)
    maxRow = (maxy - (maxy % 40)) + 40
    minCol = minx - (minx % 40)
    maxCol = (maxx - (maxx % 40)) + 40

    
    print(minRow,maxRow,minCol,maxCol)
    print(np.shape(particleCountList))

    hf = h5py.File(os.path.join(resultdir_path,"particleCountList"+".h5"),'w')
    hf.create_dataset('ParticleCount', data=particleCountList)
    hf.close()
    
    particleCountList = h5py.File(os.path.join(resultdir_path,"particleCountList"+".h5"), 'r')
    
    hydrodynamic_U_dataList=[]
    hydrodynamic_V_dataList=[]
    for LOMUQfolder in subfolders:
        try:
            fullpath = LOMUQfolder
            parentfolder = os.path.abspath(Path(LOMUQfolder).parent)
            key = fullpath[len(parentfolder)+1:]
            print("Split folder to '{}' -> {}".format(parentfolder, key))
            hydrodynamic_U = datadir_path +"/" + key + "/"+"hydrodynamic_U.h5"
            hydrodynamic_V = datadir_path +"/" + key + "/"+"hydrodynamic_V.h5"
            if not os.path.exists(hydrodynamic_U) or not os.path.exists(hydrodynamic_V):
                continue
            hydrodynamic_U_data = h5py.File(hydrodynamic_U , "r")
            hydrodynamic_V_data = h5py.File(hydrodynamic_V , "r")
            hu = hydrodynamic_U_data['uo'][()]
            # hu = hu[:,::-1,:]
            hv = hydrodynamic_V_data['vo'][()]
            # hv = hv[:,::-1,:]
            hydrodynamic_U_dataList.append(hu[...,minRow:maxRow,minCol:maxCol])
            hydrodynamic_V_dataList.append(hv[...,minRow:maxRow,minCol:maxCol])
        except ValueError:
            continue
    
    
    hf = h5py.File(resultdir_path+"/"+"hydrodynamic_U_dataList"+".h5",'w')
    hf.create_dataset('hydrodynamic_U', data=hydrodynamic_U_dataList)
    hf.close()
    
    hf = h5py.File(resultdir_path+"/"+"hydrodynamic_V_dataList"+".h5",'w')
    hf.create_dataset('hydrodynamic_V', data=hydrodynamic_V_dataList)
    hf.close()


