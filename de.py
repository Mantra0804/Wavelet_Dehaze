
import numpy as np
import pywt
import cv2
from math import sqrt,exp
from skimage import exposure
import numpy.matlib
"""## Functions"""

def  RF(img, sigma_s, sigma_r, joint_image, num_iterations = 3):
    I = img
 
    J = joint_image
    if (I.shape != J.shape):
        print('Input and joint images must have equal width and height.')
        
    h,w = J.shape
    num_joint_channels = 1
    # Estimate horizontal and vertical partial derivatives using finite
    # differences.
    dIcdx = np.diff(J, n=1, axis=1)
    dIcdy = np.diff(J, n=1, axis=0)
    
    dIdx = np.zeros((h,w))
    dIdy = np.zeros((h,w))
    
    # Compute the l1-norm distance of neighbor pixels.
    dIdx[:,1:] = dIdx[:,1:] + abs( dIcdx[:,:] )
    dIdy[1:,:] = dIdy[1:,:] + abs( dIcdy[:,:] )
    
    
    # Compute the derivatives of the horizontal and vertical domain transforms.
    dHdx = (1 + sigma_s/sigma_r * dIdx)
    dVdy = (1 + sigma_s/sigma_r * dIdy)
    
    # We do not integrate the domain transforms since our recursive filter
    # uses the derivatives directly.
    
    # The vertical pass is performed using a transposed image.
    dVdy = np.transpose(dVdy)
    
    N = num_iterations
    F = I
    
    sigma_H = sigma_s
    
    for i in range(num_iterations):
    
        #Compute the sigma value for this iteration (Equation 14 of our paper).
        sigma_H_i = sigma_H * sqrt(3) * 2**(N - (i + 1)) / sqrt(4**N - 1)
    
        F = TransformedDomainRecursiveFilter_Horizontal(F, dHdx, sigma_H_i)
        F = image_transpose(F)
    
        F = TransformedDomainRecursiveFilter_Horizontal(F, dVdy, sigma_H_i)
        F = image_transpose(F)
        
    
    
    return F



def TransformedDomainRecursiveFilter_Horizontal(I, D, sigma):

    # Feedback coefficient (Appendix of our paper).
    a = exp(-sqrt(2) / sigma);
    
    F = I
    V = a**D
    
    h ,w = I.shape
    
    # Left -> Right filter.
    for i in range(1,w):
         F[:,i] = F[:,i] + V[:,i] * ( F[:,i- 1] - F[:,i] )
        
    
    
    # Right -> Left filter.
    for i in reversed(range(w-1)):
        F[:,i] = F[:,i] + V[:,i+1] * ( F[:,i + 1] - F[:,i] )
    return F  

def image_transpose(I):

    h ,w = I.shape
      
    T = np.zeros((w ,h ))
    
    T[:,:] = np.transpose(I[:,:])
    return T

def recover(I,tran,A,tx=0.3):
  h,w,c = I.shape
  res = np.zeros((h, w, c))
  
  #tran = max(tran,tx)
  
  res[:,:,0] = (I[:,:,0] - A[0])/tran + A[0]
  res[:,:,1] = (I[:,:,1] - A[1])/tran + A[1]
  res[:,:,2] = (I[:,:,2] - A[2])/tran + A[2]
  return res

def boxfilter(img,r):
  r=8
  hei, wid = img.shape
  imDst = np.zeros(img.shape)

  #cumulative sum over Y axis
  imCum = img.cumsum(axis=0)
  #difference over Y axis
  imDst[0:r+1,:] = imCum[r:2*r+1,:]
  imDst[r+1:hei-r,:] = imCum[2*r+1:hei,:] - imCum[0:hei-2*r-1,:]
  imDst[hei-r:hei,:] = np.matlib.repmat(imCum[hei-1,:], r, 1) - imCum[hei-2*r-1:hei-r-1,:]
  #cumulative sum over X axis
  imCum = imDst.cumsum(axis=1)
  #difference over Y axis
  
  imDst[:,0:r+1] = imCum[:,r:2*r+1]
  imDst[:,r+1:wid-r] = imCum[:,2*r+1:wid] - imCum[:,0:wid-2*r-1]
  imDst[:,wid-r:wid] = np.transpose(np.matlib.repmat(imCum[:,wid-1], r, 1)) - imCum[:,wid-2*r:wid-r]
  
  
  return imDst

def guidedfilter(I,p,r,eps):
  hei, wid = I.shape
  N = boxfilter(np.ones((hei, wid)), int(r)) # the size of each local patch; N=(2r+1)^2 except for boundary pixels.

  mean_I = boxfilter(I, r) / N
  mean_p = boxfilter(p, r) / N
  mean_Ip = boxfilter(I*p, r) / N
  cov_Ip = mean_Ip - mean_I * mean_p # this is the covariance of (I, p) in each local patch.

  mean_II = boxfilter(I*I, r) / N
  var_I = mean_II - mean_I * mean_I

  a = cov_Ip / (var_I + eps) # Eqn. (5) in the paper;
  b = mean_p - a * mean_I # Eqn. (6) in the paper;

  mean_a = boxfilter(a, r) / N
  mean_b = boxfilter(b, r) / N

  q = mean_a * I + mean_b #Eqn. (8) in the paper;
  return q

def trans(img,airlight):
  N = 8
  omega = 0.95 # the amount of haze we're keeping
  im3 = np.zeros(img.shape)
  for i in range(3):
    im3[:,:,i]= img[:,:,i]/airlight[i]
  transmission = 1-(omega*opendarkchannel(im3,N))
  return transmission

def est_airlight(img,dark):
  dark_vec = np.ravel(dark)
  indices = np.argsort(-1*dark_vec)
  num_max = max(dark_vec.size//1000,1)
  airlight = np.zeros((1,1,3))
  imvec = np.resize(img,(img.shape[0]*img.shape[1],3))
  for ind in indices[:num_max]: 
    airlight += imvec[ind][:]
  return np.ravel(airlight/num_max)

def opendarkchannel(img,N):
    dark = np.min(img,axis=2)
    se = np.ones((N,N))
    dark = cv2.erode(dark,se)
    dark = cv2.dilate(dark,se)
    return dark

def wave_dehaze(img):
  dark = opendarkchannel(img,8)
  A = est_airlight(img,dark)
  transmission = trans(img,A)
  jointImg = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
  transmission = guidedfilter(jointImg, transmission ,np.ceil(30/4), 0.0001)
  t = RF(transmission,10,0.1,jointImg,3)
  out = recover(img, t, A)
  return out,t

def get_tau(Cr,Cg,Cb):
  coeff = []
  coeff.extend(np.ravel(Cr[2])/255)
  coeff.extend(np.ravel(Cg[2])/255)
  coeff.extend(np.ravel(Cb[2])/255)
  tau = np.median(np.abs(coeff))/0.6745
  return tau

"""## Main"""

def dehaze(img):
    b,g,r = cv2.split(img)
    Cb = pywt.wavedec2(b,wavelet='sym4',level=2)
    Cg = pywt.wavedec2(g,wavelet='sym4',level=2)
    Cr = pywt.wavedec2(r,wavelet='sym4',level=2)
    img2 = cv2.merge([Cr[0],Cg[0],Cb[0]])/255
    imgD,t = wave_dehaze(img2/4)
    tau = get_tau(Cr,Cg,Cb)

    HD,VD,DD = list(),list(),list()


    for i in range(1,3):
        
        CHDr,CVDr,CDDr = Cr[i][0],Cr[i][1],Cr[i][2]
        CHDg,CVDg,CDDg = Cg[i][0],Cg[i][1],Cg[i][2]
        CHDb,CVDb,CDDb = Cb[i][0],Cb[i][1],Cb[i][2]
        
        CHD = cv2.merge([CHDr,CHDg,CHDb])/255
        CVD = cv2.merge([CVDr,CVDg,CVDb])/255
        CDD = cv2.merge([CDDr,CDDg,CDDb])/255
        
        t = cv2.resize(t,(CHD.shape[1],CHD.shape[0]),interpolation=cv2.INTER_CUBIC)  
        td = cv2.merge([t,t,t])
        CHD = pywt.threshold(CHD,value=tau,mode='soft')              #Eqn(12)
        CVD = pywt.threshold(CVD,value=tau,mode='soft')              #Eqn(12)
        CDD = pywt.threshold(CDD,value=tau,mode='soft')              #Eqn(12)
      
        NCHD = CHD/td        #Equ(16)
        NCVD = CVD/td           #Equ(16)
        NCDD = CDD/td           #Equ(16) 
      
        HD.append(cv2.split(NCHD))
        VD.append(cv2.split(NCVD))
        DD.append(cv2.split(NCDD))

        imgDb,imgDg,imgDr = cv2.split(imgD*4)

    NAb,NAg,NAr = list(),list(),list()
    NAb.append(imgDb)
    NAg.append(imgDg)
    NAr.append(imgDr)


    for hd,vd,dd in zip(HD,VD,DD):
      NAb.append(tuple([hd[0],vd[0],dd[0]]))
      NAg.append(tuple([hd[1],vd[1],dd[1]]))
      NAr.append(tuple([hd[2],vd[2],dd[2]]))


    dr = pywt.waverec2(NAr,wavelet='sym4')
    dg = pywt.waverec2(NAg,wavelet='sym4')
    db = pywt.waverec2(NAb,wavelet='sym4')

    dehazed_image = cv2.merge((dr,dg,db))


    return dehazed_image

def adjust(img):
  
  minn = np.min(np.ravel(img))
  img = img-minn
  img=img/max(np.ravel(img))
  b,g,r = cv2.split(img)
  bp1,bp99 = np.percentile(b, (1, 99))
  gp1,gp99 = np.percentile(g, (1, 99))
  rp1,rp99 = np.percentile(r, (1, 99))
  Jr = exposure.rescale_intensity(r,in_range=(rp1,rp99))
  Jg = exposure.rescale_intensity(g,in_range=(gp1,gp99))
  Jb = exposure.rescale_intensity(b,in_range=(bp1,bp99))
  return cv2.merge((Jb,Jg,Jr))


