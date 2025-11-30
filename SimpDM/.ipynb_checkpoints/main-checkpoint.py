import torch
import numpy as np
import zero
from model import SimpDM
from load_data import make_dataset, prepare_fast_dataloader
from model.modules import MLPDiffusion
import json

import pandas as pd
import argparse
from hyperimpute.plugins.utils.metrics import RMSE
from hyperimpute.utils.metrics import generate_score

from ForestDiffusion import ForestDiffusionModel
from load_data import Dataset

class Trainer:
    def __init__(self, diffusion, train_iter, lr, weight_decay, epochs, device=torch.device('cuda:0'), data=None):
        self.diffusion = diffusion
        self.train_iter = train_iter
        self.epochs = epochs
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(self.diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.log_every = 100
        self.print_every = 1000
        self.ema_every = 1000
        self.data = data

    def _anneal_lr(self, step):
        frac_done = step / self.epochs
        lr = self.init_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _run_step(self, x, mask):
        x = x.to(self.device)
        mask = mask.to(self.device)

        self.optimizer.zero_grad()
        loss_gauss, loss_ssl = self.diffusion.train_iter(x, mask)
        loss = loss_gauss + loss_ssl
        loss.backward()
        self.optimizer.step()

        return loss_gauss, loss_ssl

    def run_loop(self):
        step = 0
        curr_loss_gauss = 0.0
        curr_loss_ssl = 0.0

        curr_count = 0
        while step < self.epochs:
            x, mask = next(self.train_iter)
            batch_loss_gauss, batch_loss_ssl = self._run_step(x, mask)

            self._anneal_lr(step)

            curr_count += len(x)
            curr_loss_gauss += batch_loss_gauss.item() * len(x)
            curr_loss_ssl += batch_loss_ssl.item() * len(x)

            if (step + 1) % self.log_every == 0:
                gloss = curr_loss_gauss / curr_count
                ssl_loss = curr_loss_ssl / curr_count
                if (step + 1) % self.print_every == 0:
                    print('Step {}/{}  DM Loss: {:.6f} SSL Loss:{:.6f}, Sum: {:.6f}'
                          .format((step + 1), self.epochs, gloss, ssl_loss, gloss + ssl_loss))

                curr_count = 0
                curr_loss_gauss = 0.0
                curr_loss_ssl = 0.0

            step += 1

def summarize_results(results, args):
    final_result = {}
    all_result = {}
    for key in results[0]:
        rmses = []
        for trial in range(args.n_trial):
            rmses.append(results[trial][key])
        rmse_mean, rmse_std = generate_score(rmses)
        final_result[key] = '{:.4f}+-{:.4f}'.format(rmse_mean, rmse_std)
        all_result[key] = rmses
        print('{}: {}'.format(key, final_result[key]))
    return final_result

def stratified_sample(X, y, n_per_class):
    classes = np.unique(y)
    idxs = []

    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) < n_per_class:
            raise ValueError(f"Class {c} has only {len(c_idx)} samples.")
        chosen = np.random.choice(c_idx, size=n_per_class, replace=False)
        idxs.append(chosen)

    idxs = np.concatenate(idxs)
    return idxs

def main(args, device = torch.device('cuda:0'), seed = 0):

    ####################### LOAD DATA #######################
    zero.improve_reproducibility(seed)

    # preset dataset
    D_full, y_full, data_mean, data_std = make_dataset(args) ## ADDED: save mean and std from ORIGINAL RAW DATASET
    
    if args.subsample_size:
        # sample per class
        # idx = stratified_sample(D.X_num['x_miss'], y, n_per_class=20)
        
        # sample randomly
        idx = np.random.choice(range(D_full.X_num['x_miss'].shape[0]), size=args.subsample_size, replace=False)
        x_miss_sub = D_full.X_num['x_miss'][idx]
        x_gt_sub   = D_full.X_num['x_gt'][idx]
        mask_sub   = D_full.X_num['miss_mask'][idx]
        y_sub      = pd.DataFrame(y_full.iloc[idx])
        y_sample_fact, y_cats = pd.factorize(y_sub.squeeze())
    
        D = Dataset(
            X_num={
                'x_miss': x_miss_sub,
                'x_gt': x_gt_sub,
                'miss_mask': mask_sub,
            },
            X_cat=None
        )
    else:
        D = D_full
        y = y_full
    num_numerical_features = D.X_num['x_miss'].shape[1]
    d_in = num_numerical_features
    d_out = num_numerical_features
    
    model = MLPDiffusion(d_in=d_in, d_out=d_out, d_layers=[args.hidden_units] * args.num_layers)
    model.to(device)

    ####################### TRAIN #######################
    train_loader = prepare_fast_dataloader(D, split='train', batch_size=args.batch_size)

    diffusion = SimpDM(num_numerical_features=num_numerical_features, denoise_fn=model, device=device,
                       num_timesteps=args.num_timesteps, gammas=args.gammas, ssl_loss_weight=args.ssl_loss_weight)
    diffusion.to(device)
    diffusion.train()
    trainer = Trainer(diffusion, train_loader, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs,
                      device=device, data=D) #ADDED: training function
    trainer.run_loop()

    ####################### IMPUTE #######################
    diffusion.eval()
    
    # Version for SimpDM (NaNs replaced)
    X_simpdm = torch.from_numpy(D.X_num['x_miss']).float()
    X_simpdm = torch.nan_to_num(X_simpdm, nan=-1)
    mask = torch.from_numpy(D.X_num['miss_mask']).float()
    
    x_imputed = diffusion.impute(X_simpdm.to(device), mask.to(device))
    x_imputed_unreg = x_imputed * data_std + data_mean
    ####################### EVALUATE #######################
    result = {}
    
    x_test_gt = D.X_num['x_gt'] 
    mask_np   = D.X_num['miss_mask']
    x_test_gt_unreg = x_test_gt * data_std + data_mean
    rmse = RMSE(x_imputed, x_test_gt, mask_np)
    rmse2 = RMSE(x_imputed_unreg, x_test_gt_unreg, mask_np)
    result['rmse'] = rmse
    result['rmse_unreg'] = rmse2
    print("SimpDM finished. RMSE on model subspace:", rmse)
    print("RMSE on original scale:", rmse2)
    
    # forest diffusion
    if args.fdiff:
        X_forest = D.X_num['x_miss'].astype(float)  
        Xy = np.concatenate([X_forest, y_sample_fact[:, None]], axis=1)
        
        forest_model = ForestDiffusionModel(
            Xy,
            n_t=10,
            duplicate_K=1,
            bin_indexes=[],
            cat_indexes=[Xy.shape[1] - 1],
            int_indexes=[],
            diffusion_type='vp',
            n_jobs=1,
            max_depth=3,
            n_estimators=20
        )
        
        # Fast impute
        Xy_fake = forest_model.impute(k=1)
        fd_rmse = RMSE(Xy_fake[:, :-1], x_test_gt, mask_np)
        result['fd_rmse'] = fd_rmse
        print("Forest diffusion finished. RMSE:", fd_rmse)
        
        # Repaint impute
        Xy_fake_slow = forest_model.impute(repaint=True, r=10, j=2, k=1)
        fd_rmse_slow = RMSE(Xy_fake_slow[:, :-1], x_test_gt, mask_np)
        result['fd_rmse_repaint'] = fd_rmse_slow
        print("Forest diffusion (repaint) finished. RMSE:", fd_rmse_slow)
    
        return result, Xy, x_imputed, x_imputed_unreg, Xy_fake, Xy_fake_slow, x_test_gt
    
    else:
        return result, x_imputed, x_imputed_unreg, x_test_gt, mask_np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # exp param
    parser.add_argument("--n_trial", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")

    # data param
    parser.add_argument("--dataset", type=str, default="iris",
                        choices=["iris", "yacht", "housing", "diabetes", "blood", "energy", "german", "concrete", "yeast",
                                "airfoil", "wine_red", "abalone", "wine_white", "phoneme", "power", "ecommerce",
                                 "california", "unbalanced"])
    parser.add_argument("--scenario", type=str, default="MCAR")
    parser.add_argument("--missing_ratio", type=float, default=0.3)

    # training params
    parser.add_argument("--epochs", type=int, default=10000) # 10000
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=4096)

    # model params
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_units", type=int, default=256)
    parser.add_argument("--num_timesteps", type=int, default=10)
    parser.add_argument("--ssl_loss_weight", type=float, default=1)
    parser.add_argument("--gammas", type=str, default="1_0.8_0.001")
    
    ## ADDED: add own dataset, add subsample size (forest diffusion cant run on huge dataset)
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--subsample_size", type=int, default=None) # 175341 is full size for unbalanced

    parser.add_argument("--fdiff", type=bool, default=False)
    
    args = parser.parse_args()
    device = torch.device(args.device)

    args.gammas = args.gammas.split('_')
    args.gammas = [float(gamma) for gamma in args.gammas]

    timer = zero.Timer()
    timer.run()

    results = []
    # simpdm train and impute
    for trial in range(args.n_trial):
        if args.fdiff:
            result, Xy, x_imputed, x_imputed_unreg, Xy_fake, Xy_fake_slow, x_test_gt = \
            main(seed=trial, device=device, args=args)
        else:
            result, x_imputed, x_imputed_unreg, x_test_gt, mask_np = \
            main(seed=trial, device=device, args=args)
        results.append(result)
        #ADDED: save imputed data
        np.savetxt(f'SimpDM_imputed_{args.dataset}_misprop_{args.missing_ratio}_trial{trial}.csv', 
                   x_imputed, delimiter=',')
        np.savetxt(f'SimpDM_imputed_unreg_{args.dataset}_misprop_{args.missing_ratio}_trial{trial}.csv', 
                   x_imputed_unreg, delimiter=',')
        np.savetxt(f"test_gt_{args.dataset}_misprop_{args.missing_ratio}_imbalanced_trial{trial}.csv",
                   x_test_gt, delimiter=',')
        np.savetxt(f"mask_{args.dataset}_misprop_{args.missing_ratio}_imbalanced_trial{trial}.csv",
                   mask_np, delimiter=',')
        if args.fdiff:
            np.savetxt(f"original_data_misprop_{args.missing_ratio}_imbalanced_trial{trial}.csv",
                   Xy, delimiter=',')
            np.savetxt(f"ForestDiff_fast_imputed_misprop_{args.missing_ratio}_imbalanced_trial{trial}.csv",
                       Xy_fake, delimiter=',')
            np.savetxt(f"ForestDiff_slow_imputed_misprop_{args.missing_ratio}_imbalanced_trial{trial}.csv",
                       Xy_fake_slow, delimiter=',')

    final_results = summarize_results(results, args)
    
    with open(f"SimpDM_imputed_full_{args.dataset}_misprop_{args.missing_ratio}_rmse_dict.json", 'w') as f:
        json.dump(final_results, f)
        


