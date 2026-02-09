import numpy as np
import tensorflow as tf
from tqdm import tqdm
from numpy import linalg as LA
from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.utils.tf_utils import process_grad
import time

from sklearn.metrics import pairwise_distances

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);
        
        
        
        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.client_model)
        if self.dataset == "cifar10_shard":
            MAX_CLIENTS = 500
            self.clients = self.clients[:MAX_CLIENTS]
        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = self.client_model.get_params()

        # initialize system metrics
        self.metrics = Metrics(self.clients, params)
        
        self.norm_diff = np.zeros((len(self.clients), len(self.clients)))
        self.norm_diff2 = np.zeros((len(self.clients), len(self.clients))) 


    








    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
        return all_clients


    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
            
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]

        return ids, groups, num_samples, tot_correct, losses


    def show_grads(self):
        '''
        Return:
            gradients on all workers and the global gradient
        '''

        global_grads = None
        samples = []
        cc = 0

        self.client_model.set_params(self.latest_model)

        for c in self.clients:
            num_samples, client_grads = c.get_grads(None)  # model_len unused now
            samples.append(num_samples)

            if global_grads is None:
                global_grads = np.zeros_like(client_grads)
                intermediate_grads = np.zeros((len(self.clients) + 1, client_grads.size))

            global_grads += client_grads * num_samples
            intermediate_grads[cc] = client_grads
            cc += 1

        global_grads = global_grads / np.sum(samples)
        intermediate_grads[-1] = global_grads

        return intermediate_grads


 
  
    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        return ids, groups, num_samples, tot_correct

    def save(self):
        pass
    
    #def select_cl_submod(self, round, num_clients, N_i, True ):
    def select_cl_submod(self, round, num_clients, N_i,reward,prev_grad_sum, stochastic_greedy ):
        print ("entered in cl_submod")
        
        

        if stochastic_greedy:
            
            SUi,N_i,reward, prev_grad_sum = self.stochastic_greedy(num_clients,0.1,N_i,reward, prev_grad_sum, round )
            
        else:
            SUi = self.lazy_greedy(num_clients)
        # print('Set Diff:', SUi0.difference(SUi), SUi.difference(SUi0))
                
        indices = np.array(list(SUi))
        selected_clients = np.asarray(self.clients)[indices]
        
        
       
        return indices, selected_clients,self.all_grads, N_i, reward, prev_grad_sum
    



#####################################################################
    def stochastic_greedy(self, num_clients, subsample, N_i, reward, prev_grad_sum, round):
        print("\n================ DivFedMAB=================")
        print(f"Round t = {round}")

        N = len(self.clients)
        print(f"Number of clients N = {N}")

        # --- temporary storage for this round ---
        r_norm_round = np.zeros(N)
        mu_round = np.zeros(N)


        # for k in range(N):
        #     print(f"k={k} :", self.norm_diff[k])


        if round == 0:
            print("\n--- ROUND 0: select all clients ---")
            S = set(range(N))
            for k in range(N):
                N_i[k] = 1

            self.r_norm_history[round, :] = 0.0


            return S, N_i, reward, prev_grad_sum

        # ---------- Algorithm 1: DivFedMAB ----------
        S = []                     # selected set
        V = list(range(N))         # remaining candidates


        # ---- Greedy selection of num_clients ----
        for sel_idx in range(num_clients):
            # print("\n----------------------------------")
            # print(f"Greedy selection step {sel_idx + 1}")

            r_vals = {}    # r_{t+1}^l
            r_norm = {}    # r'_{t+1}^l
            mu_new = {}    # mu_hat_{t+1}^l
            lcb = {}       # LCB values

            # ----- Lines 7–14: marginal distance -----
            for l in V:
                # print(f"\nEvaluating candidate l = {l}")

                if len(S) == 0:
                    d_prime = self.norm_diff[:, l]

                else:
                    d_to_S = np.min(self.norm_diff[:, S], axis=1)
                    d_to_l = self.norm_diff[:, l]
                    d_prime = np.minimum(d_to_S, d_to_l)


                r_vals[l] = np.sum(d_prime)
                # print(f"r_vals[{l}] = sum(d_prime) = {r_vals[l]}")

            # ----- Line 15: normalization -----
            max_r = max(r_vals.values())

            for l in V:
                r_norm_val = r_vals[l] / max_r if max_r > 0 else 0.0
                r_norm[l] = r_norm_val
                r_norm_round[l] = r_norm_val




            # ----- Lines 16–18: reward update + LCB -----
            # print("\nReward update and LCB computation")
            for l in V:
                mu_new[l] = (round * reward[l] + r_norm[l]) / (round + 1)
                mu_round[l] = mu_new[l]   # <-- STORE IT

                if N_i[l] > 0:
                    bonus = np.sqrt((2 * np.log(round)) / N_i[l])
                else:
                    bonus = float("inf")

                lcb[l] = mu_new[l] - bonus



            # ----- Line 20: select client with minimum LCB -----
            i_star = min(lcb, key=lcb.get)

            # ----- Line 22: update sets and counters -----
            S.append(i_star)
            V.remove(i_star)
            N_i[i_star] += 1
            reward[i_star] = mu_new[i_star]

  

    
        # ---- save r_norm for this round (server-level storage) ----
        self.r_norm_history[round, :] = r_norm_round
        self.mu_history[round, :] = mu_round


        print("\n=========== END DivFedMAB ===========")
        return set(S), N_i, reward, prev_grad_sum




      
#############################################################################################################################################
    # def stochastic_greedy(self, num_clients, subsample, N_i, reward, prev_grad_sum, round):
    #     print ("Running correct")
    #     N = len(self.clients)

    # # ---------- ROUND 0 (Algorithm 2 ) ----------
    #     if round == 0:
    #         S = set(range(N))          # select all clients
    #         for k in range(N):
    #             N_i[k] = 1             # N_0^k = 1
    #         return S, N_i, reward, prev_grad_sum

    #     # ---------- Algorithm 1: DivFedMAB ----------
    #     S = []                         # S_{t+1} ← ∅
    #     V = list(range(N))             # V ← {1,...,N}

    #     for _ in range(num_clients):   # for i = 1 → K

    #         r_vals = {}    # r_{t+1}^l
    #         r_norm = {}    # r'_{t+1}^l
    #         mu_new = {}    # μ̂_{t+1}^l
    #         lcb = {}       # μ̂_{t+1}^{l,-}

    #         # ----- Lines 7–14: marginal distance -----
    #         for l in V:
    #             if len(S) == 0:
    #                 d_prime = self.norm_diff[:, l]
    #             else:
    #                 d_prime = np.minimum(
    #                     np.min(self.norm_diff[:, S], axis=1),
    #                     self.norm_diff[:, l]
    #                 )

    #             r_vals[l] = np.sum(d_prime)

    #         # ----- Line 15: normalization -----
    #         max_r = max(r_vals.values())
    #         for l in V:
    #             r_norm[l] = r_vals[l] / max_r if max_r > 0 else 0.0

    #         # ----- Lines 16–18: reward + LCB -----
    #         for l in V:
    #             mu_new[l] = (round * reward[l] + r_norm[l]) / (round + 1)

    #             if N_i[l] > 0:
    #                 lcb[l] = mu_new[l] - np.sqrt((2 * np.log(round)) / N_i[l])
    #             else:
    #                 lcb[l] = -np.inf   # force exploration

    #         # ----- Line 20: select client with min LCB -----
    #         i_star = min(lcb, key=lcb.get)

    #         # ----- Line 22: update sets and counters -----
    #         S.append(i_star)
    #         V.remove(i_star)
    #         N_i[i_star] += 1
    #         reward[i_star] = mu_new[i_star]

    #     return set(S), N_i, reward, prev_grad_sum




########################################################################################################################################################
#     def stochastic_greedy(self, num_clients, subsample,N_i,reward,prev_grad_sum, round ):           #FedMAB method
        
        
#         V_set = set(range(len(self.clients)))
       
#         #reward_curr =  np.array( [1e-20] * len(self.clients) )
#         reward_curr =  np.zeros(len(self.clients))
#         lcb =  np.zeros(len(self.clients))
#         marg_util_normalized = np.zeros(len(self.clients))
#         #print ( "reward_curr before starting the loop :",  reward_curr)
#         #print (V_set)
       
#         SUi = set()
#         prev_selected = []
#         V_set = list(V_set)
        
#         val = []
        
#         #if round == 0:
#             #self.client_rewards = {i: [] for i in range(len(self.clients))}
     
#         for idx in range (len(self.clients)):
#             if N_i[idx] == 0:
#                 val.append(0)
# #            val.append (N_i[idx])
            
#             else:
#                 value = np.sqrt( (2* np.log(round) )  / (N_i[idx]) )
#                 val.append (value)
#         val=np.array(val)
        
#         if round == 0:
#             SUi = V_set
            
#             for idx in range (len(self.clients)):
#                 N_i[ idx ] = N_i[ idx] + 1
                
            
   
#         else:
            
#             for ni in range(num_clients):
#                 if ni == 0 :
                    
#                     #print ("we are selecting client", ni+1)
#                     #print ("reward till now is ", reward)
#                     marg_util = (self.norm_diff[:, V_set].sum(0))
#                     #print ("marg_util matrix sum is :", marg_util)
#                     max_marg_util = marg_util.max()
#                     marg_util_normalized  = marg_util / max_marg_util
#                     #print ("marg_util matrix sum after dividing by max value is :", marg_util_normalized)
                    
#                     #print ("v_set is", V_set)
                    
#                     reward_curr [V_set] = (((round-1)* reward [V_set] + marg_util_normalized) / round)
#                     #print ("reward_curr [V_set] when i am selecting first client :", reward_curr [V_set])
                    
#                     lcb[V_set] = reward_curr[V_set] - val[V_set]
#                     #print ("lcb for client is" ,lcb)
                    
#                     i =lcb[V_set].argmin()
#                     #print ("selected index is ",  V_set [i])
#                     client_min = self.norm_diff[:, V_set[i]]
#                     prev_selected.append(client_min)
#                     #print ("previous selected is", prev_selected )
                   
#                 else:
#                     #print ("Selecting client : ", ni+1)
#                     #print ("previously selected", prev_selected )
#                     ans = np.min (prev_selected, axis = 0)
#                     #print ("minimum of prev selected", ans )
                    
     
#                     client_min_R = np.minimum(ans[:,None], self.norm_diff[:,V_set])
#                     #print ("minimum of previously selected and current unselected :",   )

#                     marg_util = client_min_R.sum(0) 
                    
#                     #print ("column sum of the matrix ", marg_util)
#                     max_marg_util = marg_util.max()
#                     marg_util_normalized = marg_util / max_marg_util
#                     #print ("marg_util matrix sum after dividing by max value is :", marg_util_normalized)
                    
#                     #print ("v_set is", V_set)
#                     reward_curr [V_set]  = (((round-1)* reward[V_set]  +  marg_util_normalized )/round)
#                     #print ("the reward for the unselected clients ", reward_curr [V_set])
                    

#                     lcb [V_set] = reward_curr[V_set] - val[V_set]
#                     #print ("The lcb term will be", lcb)
                    
#                     #print ("lcb is ", lcb)
                    
                    
#                     i =lcb[V_set].argmin()
#                     #print ("selected index is ", V_set [i])
#                     client_min = client_min_R[:, i]
#                     prev_selected.append(client_min_R[:, i])
#                     #print ("current reward is", reward_curr [V_set] )
                    
                    
                
#                 N_i[V_set[i]] = N_i[V_set[i]] + 1
#                 SUi.add(V_set[i])
#                 V_set.remove(V_set[i])
    
           
#             reward= reward_curr
#             #print ("The reward of clients of this round is", reward)
#             #print('Number of times client get selected is as as follows ',N_i)
        


#         return SUi, N_i, reward, prev_grad_sum
        
    
        
        
            
            
#####################################################################################################################################################     
    
        

    def greedy(self, num_clients):
        # initialize the ground set and the selected set
        print('entered  to greedy ')
        V_set = set(range(len(self.clients)))
        SUi = set()
        for ni in range(num_clients):
            R_set = list(V_set)
            if ni == 0:
                marg_util = self.norm_diff[:, R_set].sum(0)
                i = marg_util.argmin()
                client_min = self.norm_diff[:, R_set[i]]
            else:
                client_min_R = np.minimum(client_min[:,None], self.norm_diff[:,R_set])
                marg_util = client_min_R.sum(0)
                i = marg_util.argmin()
                client_min = client_min_R[:, i]
            # print(R_set[i], marg_util[i])
            SUi.add(R_set[i])
            V_set.remove(R_set[i])
        return SUi

    def lazy_greedy(self, num_clients):
        # initialize the ground set and the selected set
        V_set = set(range(len(self.clients)))
        SUi = set()

        S_util = 0
        marg_util = self.norm_diff.sum(0)
        i = marg_util.argmin()
        L_s0 = 2. * marg_util.max()
        marg_util = L_s0 - marg_util
        client_min = self.norm_diff[:,i]
        # print(i)
        SUi.add(i)
        V_set.remove(i)
        S_util = marg_util[i]
        marg_util[i] = -1.
        
        while len(SUi) < num_clients:
            argsort_V = np.argsort(marg_util)[len(SUi):]
            for ni in range(len(argsort_V)):
                i = argsort_V[-ni-1]
                SUi.add(i)
                client_min_i = np.minimum(client_min, self.norm_diff[:,i])
                SUi_util = L_s0 - client_min_i.sum()

                marg_util[i] = SUi_util - S_util
                if ni > 0:
                    if marg_util[i] < marg_util[pre_i]:
                        if ni == len(argsort_V) - 1 or marg_util[pre_i] >= marg_util[argsort_V[-ni-2]]:
                            S_util += marg_util[pre_i]
                            # print(pre_i, L_s0 - S_util)
                            SUi.remove(i)
                            SUi.add(pre_i)
                            V_set.remove(pre_i)
                            marg_util[pre_i] = -1.
                            client_min = client_min_pre_i.copy()
                            break
                        else:
                            SUi.remove(i)
                    else:
                        if ni == len(argsort_V) - 1 or marg_util[i] >= marg_util[argsort_V[-ni-2]]:
                            S_util = SUi_util
                            # print(i, L_s0 - S_util)
                            V_set.remove(i)
                            marg_util[i] = -1.
                            client_min = client_min_i.copy()
                            break
                        else:
                            pre_i = i
                            SUi.remove(i)
                            client_min_pre_i = client_min_i.copy()
                else:
                    if marg_util[i] >= marg_util[argsort_V[-ni-2]]:
                        S_util = SUi_util
                        # print(i, L_s0 - S_util)
                        V_set.remove(i)
                        marg_util[i] = -1.
                        client_min = client_min_i.copy()
                        break
                    else:
                        pre_i = i
                        SUi.remove(i)
                        client_min_pre_i = client_min_i.copy()
        return SUi

    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''

        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return indices, np.asarray(self.clients)[indices]

    def aggregate(self, wsolns):
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])

        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w*v.astype(np.float64)
    
        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

    def aggregate_simple(self, wsolns):
        total_weight = 0.0
        base = [0]*len(wsolns[0][1])

        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += 1
            for i, v in enumerate(soln):
                base[i] += v.astype(np.float64)
    
        averaged_soln = [v / total_weight for v in base]

        return averaged_soln
    
    def aggregate_submod(self, wsolns, gammas):
        total_weight = 0.0
        total_gamma = 0.0
        base = [0]*len(wsolns[0][1])
        
        gammas = list(gammas)
        for (wsols, gamma) in zip(wsolns, gammas):
            total_weight += wsols[0]
            for i, v in enumerate(wsols[1]):
                base[i] += gamma*wsols[0]*v.astype(np.float64)
            total_gamma +=gamma
    
        averaged_soln = [v / (total_weight*total_gamma) for v in base]

        return averaged_soln

