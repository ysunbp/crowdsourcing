import csv
import os
from charset_normalizer import detect
import numpy as np
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch
from re import template
from scipy.special import gamma
from numpy.linalg import *
from torch.linalg import det, inv
from tqdm import trange
from scipy.stats import truncnorm
from sklearn import linear_model
from scipy.stats import multivariate_normal
from scipy.integrate import quad
from scipy.optimize import leastsq
from scipy.optimize import least_squares
import scipy
from tqdm import tqdm
import heapq
import random
import math



path = './data/real_data.csv'

elephant = [[],[],[],[]]
plane = [[],[],[],[]]
fish = [[],[],[],[]]
flower = [[] for i in range(7)]
flower_learning_re = [[] for i in range(7)]
flower_learning = [[] for j in range(7)]

with open(path, 'r') as f: 
    for i, line in enumerate(f):
        if i>5:
            elephant[0].append(line.split(',')[17:22])
            elephant[1].append(line.split(',')[27:32])
            elephant[2].append(line.split(',')[32:37])
            elephant[3].append(line.split(',')[42:47])
            
            plane[0].append(line.split(',')[47:52])
            plane[1].append(line.split(',')[57:62])
            plane[2].append(line.split(',')[62:67])
            plane[3].append(line.split(',')[72:77])

            fish[0].append(line.split(',')[77:82])
            fish[1].append(line.split(',')[87:92])
            fish[2].append(line.split(',')[92:97])
            fish[3].append(line.split(',')[102:107])
            
            for j in range(7):
                flower_learning[j].append(line.split(',')[107+j*30:107+j*30+10])
                flower_learning_re[j].append(line.split(',')[117+j*30:117+j*30+10])
                flower[j].append(line.split(',')[127+j*30:127+j*30+10])

elephant = np.array(elephant)
elephant[elephant=='Yes']=1
elephant[elephant=='No']=0

plane = np.array(plane)
plane[plane=='Yes']=1
plane[plane=='No']=0

fish = np.array(fish)
fish[fish=='Yes']=1
fish[fish=='No']=0

flower = np.array(flower)
flower[flower=='Yes']=1
flower[flower=='No']=0

flower_learning = np.array(flower_learning)
flower_learning[flower_learning=='Yes']=1
flower_learning[flower_learning=='No']=0

flower_learning_re = np.array(flower_learning_re)
flower_learning_re[flower_learning_re=='Yes']=1
flower_learning_re[flower_learning_re=='No']=0


elephant_gt =   [[1,1,0,0,0],[1,1,0,0,1],[1,1,0,1,0],[0,0,1,0,1]]
plane_gt =      [[1,0,0,1,1],[1,0,1,0,0],[1,0,0,0,1],[0,1,0,1,1]]
fish_gt =       [[1,1,0,1,0],[0,0,1,0,1],[0,1,0,0,1],[1,0,1,0,1]]

flower_gt =     [[0,1,0,0,1,1,0,1,0,1],
                 [1,0,0,1,1,1,0,0,1,0],
                 [0,1,0,0,1,1,0,1,0,1],
                 [1,1,0,0,1,0,1,1,0,0],
                 [0,0,1,1,0,1,0,1,0,1],
                 [1,0,1,0,1,0,1,0,1,0],
                 [1,0,0,1,1,0,1,0,0,1]]

flower_learning_gt = [[1,0,0,1,1,0,1,1,0,0],
                      [0,0,1,1,0,1,0,1,1,0],
                      [1,0,1,0,0,1,1,0,0,1], 
                      [0,1,0,1,0,1,1,0,1,0],
                      [1,0,1,0,1,0,0,1,1,0],
                      [1,1,0,0,1,0,1,0,1,0], 
                      [1,0,0,1,0,0,1,1,0,1]]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_scores(set_name, set_gt):
    batch_results = []
    for round in range(len(set_name)):
        answers = set_name[round]
        batch_results.append([])
        for j, worker in enumerate(answers):
            correct = 0
            total = 0
            for question_idx in range(len(worker)):
                if int(worker[question_idx]) == set_gt[round][question_idx]:
                    correct += 1
                total += 1
            batch_results[round].append(correct/total)
    return batch_results


flower_results = np.array(compute_scores(flower, flower_gt))
elephant_results = np.array(compute_scores(elephant, elephant_gt))
fish_results = np.array(compute_scores(fish, fish_gt))
plane_results = np.array(compute_scores(plane, plane_gt))

def compute_prior_distribution(prior_result): 
    prior_last = (np.array(prior_result[0])+np.array(prior_result[2]))/2 # The average of two learning rounds of prior domains
    miu = np.mean(prior_last, 0)
    sigma = np.std(prior_last)
    return miu, sigma

def compute_cs_ws(target_results):
    css = [[],[],[]]
    wss = [[],[],[]]
    for i, item in enumerate(target_results):
        if i == 0:
            css[0] = item*10
        elif i == 1 or i == 2:
            if i == 1:
                css[1] = item*10
            else:
                css[1] += item*10
        else:
            if i == 3:
                css[2] = item*10
            else:
                css[2] += item*10
    wss[0] = 10-css[0]
    wss[1] = 20-css[1]
    wss[2] = 40-css[2]
    return css, wss

def initialize_cov_miu(prior_results, tolerant=0.1):
    mus = []
    sigmas = []
    rhos = []
    for prior_result in prior_results:
        miu, sigma = compute_prior_distribution(prior_result)
        mus.append(miu)
        sigmas.append(sigma)
    mus.append(np.random.uniform(0.5-tolerant,0.5+tolerant))
    sigmas.append(np.mean(sigmas))

    for i in range(6):
        rhos.append(np.random.uniform(0,1))
    cov12 = rhos[0]*sigmas[0]*sigmas[1]
    cov13 = rhos[1]*sigmas[0]*sigmas[2]
    cov14 = rhos[2]*sigmas[0]*sigmas[3]
    cov23 = rhos[3]*sigmas[1]*sigmas[2]
    cov24 = rhos[4]*sigmas[1]*sigmas[3]
    cov34 = rhos[5]*sigmas[2]*sigmas[3]
    cov = [
                [sigmas[0]**2, cov12, cov13, cov14],
                [cov12, sigmas[1]**2, cov23, cov24],
                [cov13, cov23, sigmas[2]**2, cov34],
                [cov14, cov24, cov34, sigmas[3]**2]
            ]
    eig , _ = np.linalg.eig(cov) 
    while eig[0] < 0 or eig[1] < 0 or eig[2] < 0 or eig[3] < 0 :
            
        rhos = []
        for i in range(6):
            rhos.append(np.random.uniform(0,1))
        cov12 = rhos[0]*sigmas[0]*sigmas[1]
        cov13 = rhos[1]*sigmas[0]*sigmas[2]
        cov14 = rhos[2]*sigmas[0]*sigmas[3]
        cov23 = rhos[3]*sigmas[1]*sigmas[2]
        cov24 = rhos[4]*sigmas[1]*sigmas[3]
        cov34 = rhos[5]*sigmas[2]*sigmas[3]

        cov = [
                    [sigmas[0]**2, cov12, cov13, cov14],
                    [cov12, sigmas[1]**2, cov23, cov24],
                    [cov13, cov23, sigmas[2]**2, cov34],
                    [cov14, cov24, cov34, sigmas[3]**2]
                ]

        eig , _ = np.linalg.eig(cov)
    return mus, cov, sigmas, rhos

def compute_oracle(worker, covariance, mus, m=1):
    p1, p2, p3 = worker[0], worker[1], worker[2]
    full_inv = np.linalg.inv(covariance)
    partial_cov = covariance[:3,:3]
    
    mu1, mu2, mu3, mu4 = mus[0], mus[1], mus[2], mus[3]
    
    L33 = full_inv[:3,:3]
    L11 = full_inv[3,3]
    L13 = full_inv[3,:3]
    
    p13 = np.array([p1-mu1, p2-mu2, p3-mu3]).reshape(3,1)
    A = L11*mu4*mu4+np.matmul(np.matmul(p13.T, L33), p13)-2*np.matmul(L13,p13)*mu4
    B = 2*np.matmul(L13,p13)-2*mu4*L11
    C = L11
    full_det = np.linalg.det(covariance)
    var = multivariate_normal(mean=[mu1,mu2,mu3], cov=partial_cov)
    C2 = var.pdf([p1,p2,p3])
    C1 = 1/((2*math.pi)**(2)*math.sqrt(full_det))
    h = lambda x : x*C1/C2*math.exp(-1/2*(C*x**2+B*x+A))
    result = quad(h,0,m)[0]
    return result

def compute_MLE(prior_results, cs, ws, pi, num_epochs):
    
    mus, cov, sigmas, rhos = initialize_cov_miu(prior_results, 0)
    miu = torch.tensor(mus[3])
    sigmas = torch.tensor(sigmas)
    sigmas = torch.unsqueeze(sigmas, 1)

    rhos = torch.tensor(rhos)
    rhos = torch.unsqueeze(rhos, 1)

    pre_points = []
    post_points = []
    for i in range(3):
        pre_points.append(i/3*mus[3])
        post_points.append(mus[3]+(i/3)*(1-mus[3]))
    pre_points.append(mus[3])
    post_points.append(1)
    
    miu = Variable(miu, requires_grad=True)

    sigmas = Variable(sigmas, requires_grad = True)
    rhos = Variable(rhos, requires_grad = True)

    cov12 = rhos[0]*sigmas[0]*sigmas[1]
    cov13 = rhos[1]*sigmas[0]*sigmas[2]
    cov14 = rhos[2]*sigmas[0]*sigmas[3]
    cov23 = rhos[3]*sigmas[1]*sigmas[2]
    cov24 = rhos[4]*sigmas[1]*sigmas[3]
    cov34 = rhos[5]*sigmas[2]*sigmas[3]
    
    l1 = torch.cat([sigmas[0]**2, cov12, cov13, cov14]).reshape(1,4)
    l2 = torch.cat([cov12, sigmas[1]**2, cov23, cov24]).reshape(1,4)
    l3 = torch.cat([cov13, cov23, sigmas[2]**2, cov34]).reshape(1,4)
    l4 = torch.cat([cov14, cov24, cov34, sigmas[3]**2]).reshape(1,4)
    

    cov = torch.cat([
                    l1,l2,l3,l4
                ])

    h = torch.tensor(mus[:3]).reshape(3,1)
    lr1 = 1e-7
    lr2 = 1e-4
    step = 0
    for j in range(num_epochs):
        likelihood = 0
        for i in range(len(cs)):
            step += 1
            
            c = torch.tensor(cs[i])
            w = torch.tensor(ws[i])
            cov_det = det(cov)
            L11 = cov[3,3]
            L13 = (cov[3,:3]).reshape(1,3)
            L31 = (cov[:3,3]).reshape(3,1)
            L33 = cov[:3,:3]
            p = (torch.tensor(pi[i]).reshape(3,1)).to(torch.float32)
            p = p-h
            
            m = miu+torch.mm(torch.mm(1/2*(L13+L31.reshape(1,3)), inv(L33)),p)
            sig = L11-torch.mm(torch.mm(L13, inv(L33)),L31)
            
            temp_like = 0
            for j in range(4): # numerical integration
                if j == 0 or j == 3:
                    pre = miu/8*(pre_points[j]**c*(1-pre_points[j])**w*torch.exp(-1/2*((pre_points[j]-m)**2/sig)))
                    temp_like += pre
                    post = (1-miu)/8*(post_points[j]**c*(1-post_points[j])**w*torch.exp(-1/2*((post_points[j]-m)**2/sig)))
                    temp_like += post
                else:
                    pre = 3*miu/8*(pre_points[j]**c*(1-pre_points[j])**w*torch.exp(-1/2*((pre_points[j]-m)**2/sig)))
                    temp_like += pre
                    post = 3*(1-miu)/8*(post_points[j]**c*(1-post_points[j])**w*torch.exp(-1/2*((post_points[j]-m)**2/sig)))
                    temp_like += post
            
            fac = 1
            temp_like *= fac

            likelihood -= (-torch.log(cov_det)/2+torch.log(temp_like)+1*torch.sum(1/(1-torch.abs(rhos)))) # add normalization
            
            if step % 5 == 0:
                likelihood.backward(retain_graph=True)
                rhos.data -= lr1*rhos.grad.data
                sigmas.data -= lr1*sigmas.grad.data
                miu.data -= lr2*miu.grad.data
                
                sigmas.grad.data.zero_()
                rhos.grad.data.zero_()
                miu.grad.data.zero_()
                cov12 = rhos[0]*sigmas[0]*sigmas[1]
                cov13 = rhos[1]*sigmas[0]*sigmas[2]
                cov14 = rhos[2]*sigmas[0]*sigmas[3]
                cov23 = rhos[3]*sigmas[1]*sigmas[2]
                cov24 = rhos[4]*sigmas[1]*sigmas[3]
                cov34 = rhos[5]*sigmas[2]*sigmas[3]
                
                l1 = torch.cat([sigmas[0]**2, cov12, cov13, cov14]).reshape(1,4)
                l2 = torch.cat([cov12, sigmas[1]**2, cov23, cov24]).reshape(1,4)
                l3 = torch.cat([cov13, cov23, sigmas[2]**2, cov34]).reshape(1,4)
                l4 = torch.cat([cov14, cov24, cov34, sigmas[3]**2]).reshape(1,4)
                

                cov = torch.cat([l1,l2,l3,l4])  
                likelihood = 0
            
    return sigmas.data, rhos.data, miu.data

def generate_workers(prior_domains):
    workers = []
    prior_averageds = []
    for prior_domain in prior_domains:
        averaged_prior = np.mean(prior_domain, 0)
        prior_averageds.append(averaged_prior)
    total_num_workers = len(prior_averageds[0])
    for index in range(total_num_workers):
        worker = [prior_averageds[0][index],prior_averageds[1][index],prior_averageds[2][index]]
        workers.append(worker)
    return workers

def compute_learning_cross(prior_results, pis, flags):
    def Fun(p,x,flag):
        a = p
        return 1/(1+np.exp(-(a*np.log((x)+1)-flag)))
    def error (p,x,y,flag):
        return Fun(p,x,flag)-y
    pi_in = np.concatenate([prior_results, pis])
    x_prior = np.array([(i) for i in range(2)])
    x_target = np.array([((math.pow(2, i)-1)*2) for i in range(len(pis))])
    x = np.concatenate([x_prior,x_prior,x_prior,x_target])
    p0 = [0.5]
    para = least_squares(error, p0, args=(x,pi_in,flags), bounds=((-1), (1)))
    return para['x'][0]

def ME_MLE_estimate_with_learning(prior_results, workers, num_epochs, flower_learning_results, b=0):
    cur_workers = workers # workers here contains each worker's prior mean accuracy
    num_rounds = 2
    css, wss = compute_cs_ws(flower_learning_results)
    mius, _, _,_ = initialize_cov_miu(prior_results, tolerant=0)
    pi = np.array(prior_results)[:,-1].reshape(-1,3)
    prior_p1, prior_p2, prior_p3 = prior_results
    prior_p11,_,prior_p13,_ = prior_p1
    prior_p21,_,prior_p23,_ = prior_p2
    prior_p31,_,prior_p33,_ = prior_p3
    for round in range(num_rounds):
        if round == 0:
            cs = css[round]
            ws = wss[round]
        worker_scores = []    
        size_workers = len(cur_workers)
        sigmas, rhos, miu = compute_MLE(prior_results,cs, ws, pi, num_epochs)
        cov12 = rhos[0]*sigmas[0]*sigmas[1]
        cov13 = rhos[1]*sigmas[0]*sigmas[2]
        cov14 = rhos[2]*sigmas[0]*sigmas[3]
        cov23 = rhos[3]*sigmas[1]*sigmas[2]
        cov24 = rhos[4]*sigmas[1]*sigmas[3]
        cov34 = rhos[5]*sigmas[2]*sigmas[3]
        cov = torch.tensor([
                    [sigmas[0]**2, cov12, cov13, cov14],
                    [cov12, sigmas[1]**2, cov23, cov24],
                    [cov13, cov23, sigmas[2]**2, cov34],
                    [cov14, cov24, cov34, sigmas[3]**2]
                ])
        mius[3] = miu.numpy()
        for worker in cur_workers:
            cur_score = compute_oracle(worker, cov.numpy(), mius, 1)
            worker_scores.append(cur_score)
            worker.append(cur_score)
        
        updated_learning_scores = []
        cur_piss = []
        for i, worker in enumerate(cur_workers):
            cur_prior_p11 = prior_p11[i]
            cur_prior_p12 = prior_p13[i]
            cur_prior_p21 = prior_p21[i]
            cur_prior_p22 = prior_p23[i]
            cur_prior_p31 = prior_p31[i]
            cur_prior_p32 = prior_p33[i]
            prior_pis = np.array([cur_prior_p11,cur_prior_p12,cur_prior_p21,cur_prior_p22,cur_prior_p31,cur_prior_p32])
            cur_pis = np.array(worker[3:])
            cur_piss.append(cur_pis[-1])
            flags = [-0.34 for i in range(2)] + [-2.24 for i in range(2)] + [-0.01 for i in range(2)] + [b for i in range(len(cur_pis))]
            a = compute_learning_cross(prior_pis, cur_pis, flags) # pass the domain difficulty as flags
            updated_learning_score = 1/(1+math.exp(-(a*math.log(((math.pow(2, round)-1)*2+2)+1)-flags[-1]))) #+2 means step two batches away
            updated_learning_scores.append(updated_learning_score)
        max_number = heapq.nlargest(int((size_workers+1)/2), updated_learning_scores)
        updated_workers = []
        updated_prior_p11 = []
        updated_prior_p13 = []
        updated_prior_p21 = []
        updated_prior_p23 = []
        updated_prior_p31 = []
        updated_prior_p33 = []
        updated_pi = []
        updated_cs = []
        updated_ws = []
        for t in max_number:
            index = updated_learning_scores.index(t)
            updated_learning_scores[index] = 'a'
            updated_workers.append(cur_workers[index])
            updated_prior_p11.append(prior_p11[index])
            updated_prior_p13.append(prior_p13[index])
            updated_prior_p21.append(prior_p21[index])
            updated_prior_p23.append(prior_p23[index])
            updated_prior_p31.append(prior_p31[index])
            updated_prior_p33.append(prior_p33[index])
            updated_pi.append(pi[index])
            if round+1 < num_rounds:
                updated_cs.append(css[round+1][index])
                updated_ws.append(wss[round+1][index])
        cur_workers = updated_workers
        prior_p11 = updated_prior_p11
        prior_p13 = updated_prior_p13
        prior_p21 = updated_prior_p21
        prior_p23 = updated_prior_p23
        prior_p31 = updated_prior_p31
        prior_p33 = updated_prior_p33
        pi = updated_pi
        cs = updated_cs
        ws = updated_ws
    out_index = []
    target_domain = []
    for worker in cur_workers:
        target_domain.append(worker[0])
    data = []
    for worker in workers:
        data.append(worker[0])

    for wt in target_domain:
        index = data.index(wt)
        data[index] = 'a'
        out_index.append(index)
    return out_index

def compute_avg_scores(worker_scores, indices, k=7):
    out_workers = []
    for index in indices[:k]:
        out_workers.append(worker_scores[index])
    return np.mean(out_workers)

if __name__ == '__main__':
    seed = 1998
    setup_seed(seed)
    prior_results = [elephant_results, fish_results, plane_results]
    flower_learning_results = np.array(compute_scores(flower_learning, flower_learning_gt))
    workers = generate_workers(prior_results)
    num_epochs = 50
    worker_scores = np.mean(flower_results[1:3],0)
    setup_seed(seed)
    print('ground truth', np.mean(heapq.nlargest(7, worker_scores)))
    b = math.log(1/0.5-1)
    setup_seed(seed)
    scores = []
    counter = 0
    turning_flag = True
    while turning_flag == True:
        try:
            ME_MLE_learning_index = ME_MLE_estimate_with_learning(prior_results=prior_results, workers=workers, num_epochs=num_epochs, flower_learning_results=flower_learning_results, b=b) 
            score = compute_avg_scores(worker_scores, ME_MLE_learning_index)
            counter += 1
        except ValueError:
            continue
        print('ME MLE with learning scores', score)
        scores.append(score)
        if counter == 4:
            turning_flag = False
    print(np.mean(scores))

    
