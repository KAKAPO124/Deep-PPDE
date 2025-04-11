import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import math

from lib.bsde import PPDE_Heston as PPDE
from lib.options import LookbackOption, BarrierOption

def sample_x0(batch_size, device, barrier_level=None):
    sigma = 0.3
    mu = 0.08
    tau = 0.1
    z = torch.randn(batch_size, 1, device=device)
    s0 = torch.exp((mu-0.5*sigma**2)*tau + 0.3*math.sqrt(tau)*z) # lognormal
    v0 = torch.ones_like(s0) * 0.04
    x0 = torch.cat([s0, v0], 1)
    
    # 如果指定了障碍水平，则调整初始价格以确保不触碰障碍
    if barrier_level is not None:
        x0[:, 0] = torch.clamp(x0[:, 0], min=barrier_level * 1.01 if barrier_level < 1 else barrier_level * 0.99)
    
    return x0

def write(msg, logfile, pbar):
    pbar.write(msg)
    with open(logfile, "a") as f:
        f.write(msg)
        f.write("\n")

def train(T,
          n_steps,
          d,
          mu,
          vol_of_vol,
          kappa,
          theta,
          depth,
          rnn_hidden,
          ffn_hidden,
          max_updates,
          batch_size, 
          lag,
          base_dir,
          device,
          method,
          option_type="lookback",
          barrier_type=None,
          barrier_level=None):
    
    logfile = os.path.join(base_dir, "log.txt")
    ts = torch.linspace(0, T, n_steps+1, device=device)
    
    if option_type == "lookback":
        option = LookbackOption()
    elif option_type == "barrier":
        option = BarrierOption(barrier_type=barrier_type, barrier_level=barrier_level)
    else:
        raise ValueError("Unsupported option type")
    
    ppde = PPDE(d=d, mu=mu, vol_of_vol=vol_of_vol, kappa=kappa, theta=theta,
                depth=depth, rnn_hidden=rnn_hidden, ffn_hidden=ffn_hidden)
    ppde.to(device)
    optimizer = torch.optim.RMSprop(ppde.parameters(), lr=0.001)
    
    pbar = tqdm.tqdm(total=max_updates)
    losses = []
    for idx in range(max_updates):
        optimizer.zero_grad()
        x0 = sample_x0(batch_size, device, barrier_level)
        if method == "bsde":
            loss, _, _ = ppde.fbsdeint(ts=ts, x0=x0, option=option, lag=lag)
        else:
            loss, _, _ = ppde.conditional_expectation(ts=ts, x0=x0, option=option, lag=lag)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
        
        # 测试
        if idx % 10 == 0:
            with torch.no_grad():
                x0 = torch.ones(5000, d, device=device)
                x0[:, 1] = x0[:, 1] * 0.04
                if barrier_level is not None:
                    x0[:, 0] = torch.clamp(x0[:, 0], min=barrier_level * 1.01 if barrier_level < 1 else barrier_level * 0.99)
                loss, Y, payoff = ppde.fbsdeint(ts=ts, x0=x0, option=option, lag=lag)
                payoff = torch.exp(-mu * ts[-1]) * payoff.mean()
            
            pbar.update(10)
            write("loss={:.4f}, Monte Carlo price={:.4f}, predicted={:.4f}".format(loss.item(), payoff.item(), Y[0,0,0].item()), logfile, pbar)
    
    result = {"state": ppde.state_dict(),
              "loss": losses}
    torch.save(result, os.path.join(base_dir, "result.pth.tar"))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', default='./numerical_results/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--d', default=2, type=int)
    parser.add_argument('--max_updates', default=5000, type=int)
    parser.add_argument('--ffn_hidden', default=[20,20], nargs="+", type=int)
    parser.add_argument('--rnn_hidden', default=20, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--n_steps', default=100, type=int, help="number of steps in time discrretisation")
    parser.add_argument('--lag', default=10, type=int, help="lag in fine time discretisation to create coarse time discretisation")
    parser.add_argument('--mu', default=0.05, type=float, help="risk free rate")
    parser.add_argument('--vol_of_vol', default=0.05, type=float, help="vol of vol")
    parser.add_argument('--kappa', default=0.8, type=float, help="mean reverting process coef")
    parser.add_argument('--theta', default=0.3, type=float, help="mean reverting")
    parser.add_argument('--method', default="bsde", type=str, help="learning method", choices=["bsde", "orthogonal"])
    parser.add_argument('--option_type', default="lookback", type=str, help="option type", choices=["lookback", "barrier"])
    parser.add_argument('--barrier_type', default=None, type=str, help="barrier option type")
    parser.add_argument('--barrier_level', default=None, type=float, help="barrier level")

    args = parser.parse_args()
    
    assert args.d == 2, "Heston implementation is for d=2" 
    
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device = "cpu"
    
    results_path = os.path.join(args.base_dir, "Heston", args.method)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    train(T=args.T,
          n_steps=args.n_steps,
          d=args.d,
          mu=args.mu,
          vol_of_vol=args.vol_of_vol,
          kappa=args.kappa,
          theta=args.theta,
          depth=args.depth,
          rnn_hidden=args.rnn_hidden,
          
          ffn_hidden=args.ffn_hidden,
          max_updates=args.max_updates,
          batch_size=args.batch_size,
          lag=args.lag,
          base_dir=results_path,
          device=device,
          method=args.method,
          option_type=args.option_type,
          barrier_type=args.barrier_type,
          barrier_level=args.barrier_level)
