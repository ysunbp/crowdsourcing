import csv
import os
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
from scipy import stats
from scipy.special import gamma
from tqdm import trange
from scipy.stats import truncnorm
from scipy.integrate import quad
from scipy.optimize import leastsq
from scipy.optimize import least_squares
import scipy
import heapq
import random
import math
import copy

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

def compute_prior_distribution(prior_result): 
    if len(prior_result) > 1: #real data
        print(prior_result.shape, prior_result)
        prior_last = (np.array(prior_result[0])+np.array(prior_result[2]))/2
        miu = np.mean(prior_last,0)
        sigma = np.std(prior_last)
    else:
        prior_last = prior_result
        miu = np.mean(prior_last)
        sigma = np.std(prior_last)
    return miu, sigma

def compute_target_distribution(target_result):
    target = np.array(target_result[0])
    miu4 = np.mean(target,0)
    sigma4 = np.std(target)
    return miu4, sigma4

def initialize_cov_miu(prior_results, mu4, sigma4):
    mus = []
    sigmas = []
    rhos = []
    for prior_result in prior_results:
        miu, sigma = compute_prior_distribution([prior_result])
        mus.append(miu)
        sigmas.append(sigma)
    if sigma4 == 0:
        sigma4 = np.mean(sigmas)
    mus.append(mu4)
    sigmas.append(sigma4)

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

def generate_workers(mus,cov,num_workers):
    class invgamma(stats.rv_continuous):
        def _pdf(self, x,alpha,beta):
            px = (beta**alpha)/gamma(alpha)*x**(-alpha-1)*np.exp(-beta/x)
            return px
    invgamma = invgamma(name="invgamma", a=0.0)

    workers = []
    target_distribution = []
    for i in trange(num_workers):
        p = np.random.multivariate_normal(mus, cov)
        while p[0] < 0 or p[0] > 1 or p[1] < 0 or p[1] > 1 or p[2] < 0 or p[2] > 1 or p[3] < 0 or p[3] > 1:
            p = np.random.multivariate_normal(mus, cov)
        workers.append(p)
        target_distribution.append(p[3])
    return workers, target_distribution

def generate_primal_dataset(workers, questions_per_batch):
    worker_primal = []
    for worker in workers:
        p_1 = worker[0]
        p_2 = worker[1]
        p_3 = worker[2]
        p_t = worker[3]
        domains = [p_1, p_2, p_3]
        domain_observed = []
        for domain in domains:
            correct = 0
            for i in range(questions_per_batch):
                q = random.uniform(0,1)
                if q < domain:
                    correct += 1
            domain_observed.append(correct/questions_per_batch)
        correct = 0
        for i in range(2*questions_per_batch):
            q = random.uniform(0,1)
            if q < p_t:
                correct += 1
        domain_observed.append(correct/questions_per_batch/2)
        worker_primal.append(domain_observed)
    return worker_primal

def compute_a(prior_result, pis, flags):
    def Fun(p,x,flag):
        a = p
        return 1/(1+np.exp(-(a*np.log((x)+1)-flag)))
    def error (p,x,y,flag):
        return Fun(p,x,flag)-y
    pi_in = np.concatenate([prior_result, pis])
    x_prior = np.array([(i+1) for i in range(1)])
    x_target = np.array([((math.pow(2, i+1)-1))*2 for i in range(1)])
    x = np.concatenate([x_prior,x_prior,x_prior,x_target])
    p0 = [0.5]
    para = least_squares(error, p0, args=(x,pi_in,flags), bounds=((-0.5), (0.5)))
    return para['x'][0]

def generate_whole_dataset(workers, questions_per_batch, num_rounds, bs=[-0.34, -2.24, -0.01, 0]):
    learning_per_batch = 2*questions_per_batch
    primal_dataset = generate_primal_dataset(workers, questions_per_batch)
    flags = bs
    workers_whole_process_p4 = []
    last_round_predictedp4_for_gt = []
    css = [[] for i in range(int(math.pow(2,num_rounds)-1))]
    wss = [[] for i in range(int(math.pow(2,num_rounds)-1))]
    for i, worker_primal in enumerate(primal_dataset):
        p4s = []
        cur_worker = workers[i]
        cur_p4 = cur_worker[3]
        prior_result = worker_primal[:3]
        pis = [worker_primal[3]]
        css[0].append(pis[0]*learning_per_batch)
        wss[0].append(learning_per_batch*(1-pis[0]))
        cur_a = compute_a(prior_result, pis, flags)
        p4s.append(cur_p4)
        for round in range(1,num_rounds):
            round_p4 = 1/(1+math.exp(-(cur_a*math.log(2*(math.pow(2, round)-1)+2+1)-flags[-1])))
            p4s.append(round_p4)
            if round == num_rounds:
                last_round_predictedp4_for_gt.append(round_p4)
        workers_whole_process_p4.append(p4s)
    subsequent_dataset = []
    for worker_p4 in workers_whole_process_p4:
        cur_observed = []
        t = 1
        for j, cur_generate_p4 in enumerate(worker_p4[1:]):
            correct = 0
            cur_correct = 0
            for i in range(int(learning_per_batch*(math.pow(2,j+1)))):
                q = random.uniform(0,1)
                if q < cur_generate_p4:
                    correct += 1
                    cur_correct += 1
                if (i+1) % learning_per_batch == 0:
                    css[t].append(cur_correct)
                    wss[t].append(learning_per_batch-cur_correct)
                    cur_correct = 0
                    t += 1
            cur_observed.append(correct/learning_per_batch/(math.pow(2,j+1)))
        subsequent_dataset.append(cur_observed)
    return primal_dataset, subsequent_dataset, last_round_predictedp4_for_gt, css, wss

def compute_learning_cross(prior_results, pis, flags):
    def Fun(p,x,flag):
        a = p
        return 1/(1+np.exp(-(a*np.log((x)+1)-flag)))
    def error (p,x,y,flag):
        return Fun(p,x,flag)-y
    pi_in = np.concatenate([prior_results, pis])
    x_prior = np.array([(i+1) for i in range(1)])
    x_target = np.array([((math.pow(2, i+1)-1))*2 for i in range(len(pis))])
    x = np.concatenate([x_prior,x_prior,x_prior,x_target])
    p0 = [0.5]
    para = least_squares(error, p0, args=(x,pi_in,flags), bounds=((-1), (1)))
    return para['x'][0]

def compute_MLE(prior_results, cs, ws, pi, num_epochs):
    
    mus, cov, sigmas, rhos = initialize_cov_miu(prior_results, 0.5, sigma4=0)
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
    lr2 = 1e-5
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
            for j in range(4):
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

            likelihood -= (-torch.log(cov_det)/2+torch.log(temp_like)+1*torch.sum(1/(1-torch.abs(rhos))))

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
                

                cov = torch.cat([
                                l1,l2,l3,l4
                        ])
                likelihood = 0
            
    return sigmas.data, rhos.data, miu.data


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

def ME_MLE_learning(num_rounds, css, wss, num_workers, num_epochs, primal_dataset, questions_per_batch, k, b=0):
    cur_batch_idx = int(0)
    pi = np.array(primal_dataset)[:,:3]
    prior_results = pi.reshape(3,-1)
    
    cur_p4_remained = [[] for i in range(num_workers)]

    mius, _, _,_ = initialize_cov_miu(prior_results, 0.5, sigma4=0)
    start_idx = int(math.pow(2,num_rounds))-1
    gt_scores = np.mean(np.array(css)[start_idx:start_idx+2],0)
    modified_cs = []
    modified_ws = []
    for round in range(num_rounds):
        cur_num_batches = int(math.pow(2, round))
        if cur_num_batches>1:
            cs = np.sum(np.array(css)[cur_batch_idx:cur_batch_idx+cur_num_batches,:],0).reshape(-1,1)
            ws = np.sum(np.array(wss)[cur_batch_idx:cur_batch_idx+cur_num_batches,:],0).reshape(-1,1)
        else:
            cs = np.array(css)[cur_batch_idx:cur_batch_idx+cur_num_batches,:].reshape(-1,1)
            ws = np.array(wss)[cur_batch_idx:cur_batch_idx+cur_num_batches,:].reshape(-1,1)
        modified_cs.append(cs)
        modified_ws.append(ws)
    cs = modified_cs[0]
    ws = modified_ws[0]
    for round in trange(num_rounds):
        
        cur_num_workers = int(num_workers/math.pow(2, round))
        
        sigmas, rhos, miu = compute_MLE(prior_results, cs, ws, pi, num_epochs)
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
        worker_scores = []
        flags = [-0.34 for i in range(1)] + [-2.24 for i in range(1)] + [-0.01 for i in range(1)] + [b for i in range(round+1)]
        for worker_idx in range(cur_num_workers):
            cur_pi = pi[worker_idx]
            cur_score = compute_oracle(cur_pi, cov.numpy(), mius, 1)
            cur_p4_remained[worker_idx].append(cur_score)
            a = compute_learning_cross(pi[worker_idx], cur_p4_remained[worker_idx], flags)
            updated_learning_score = 1/(1+math.exp(-(a*math.log(((math.pow(2, round+1)-1)*2+2)+1)-flags[-1])))
            worker_scores.append(updated_learning_score)
        cur_index = np.argsort(worker_scores)[-int(cur_num_workers/2):]
        updated_pi = []
        updated_gt_scores = []
        updated_cs = []
        updated_ws = []
        updated_cur_p4_remained = []
        for cur_idx in cur_index:
            updated_pi.append(pi[cur_idx])
            updated_gt_scores.append(gt_scores[cur_idx])
            updated_cs.append(modified_cs[round][cur_idx])
            updated_ws.append(modified_ws[round][cur_idx])
            updated_cur_p4_remained.append(cur_p4_remained[cur_idx])
        pi = updated_pi
        gt_scores = updated_gt_scores
        cs = updated_cs
        ws = updated_ws
        cur_p4_remained = updated_cur_p4_remained
    return np.mean(gt_scores[:k])/questions_per_batch

def gt_index(num_rounds, css, num_workers, k_number):
    start_idx = int(math.pow(2,num_rounds))-1
    gts = np.array(css)[start_idx:start_idx+2]
    gt_scores = np.mean(gts,0)
    out_index = np.argsort(gt_scores)[-k_number:]
    return out_index

def compute_score(indices, css, questions_per_batch,num_rounds):
    start_idx = int(math.pow(2,num_rounds))-1
    gt_scores = np.mean(np.array(css)[start_idx:start_idx+2],0)
    k_workers = len(indices)
    total_scores = []
    for idx in indices:
        total_scores.append(gt_scores[idx])
    mean_score = np.mean(total_scores)/questions_per_batch
    return mean_score


if __name__ == '__main__':
    seed = 2014
    setup_seed(seed)
    flower_results = np.array(compute_scores(flower, flower_gt))
    elephant_results = np.array(compute_scores(elephant, elephant_gt))
    fish_results = np.array(compute_scores(fish, fish_gt))
    plane_results = np.array(compute_scores(plane, plane_gt))
    flower_learning_results = np.array(compute_scores(flower_learning, flower_learning_gt))
    questions_per_batch = 10
    num_workers = 50
    num_epochs = 50
    k = 5
    num_rounds = math.floor(np.log2(num_workers/k)+1)

    
    mu4, sigma4 = compute_target_distribution(flower_learning_results)
    prior_results = [elephant_results, fish_results, plane_results]
    mus, cov, _, _ = initialize_cov_miu(prior_results, mu4, sigma4)
    workers, _  = generate_workers(mus,cov,num_workers)
    primal_dataset, subsequent_dataset, last_round_predictedp4_for_gt, css, wss = generate_whole_dataset(workers, questions_per_batch, num_rounds, bs=[-0.34, -2.24, -0.01, 0])
    
    print('total workers', num_workers, '| num_rounds', num_rounds-1, '| k', k, '| learning_per_batch', questions_per_batch*2, '| seed', seed)
    
    scores = []
    setup_seed(seed)
    
    b = math.log(1/0.5-1)
    counter = 0
    turning_flag = True
    while turning_flag == True:
        try:
            score = ME_MLE_learning(num_rounds-1, css, wss, num_workers, num_epochs, primal_dataset, questions_per_batch*2, k, b=b)
            counter += 1
        except ValueError:
            continue
        print('ME_MLE_LEANRING', score)
        scores.append(score)
        if counter == 4:
            turning_flag = False
    print('avg score', np.mean(scores))

    gt = gt_index(num_rounds-1, css, num_workers, k)
    print('ground truth', compute_score(gt, css, questions_per_batch*2,num_rounds-1))
    print(gt)
    print('========================================================================')
