import torch
import torch.nn as nn
import numpy as np
import math
import json
import os
import time

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

N_BITS=8; DELAY=10; INPUT_DIM=9; OUTPUT_DIM=8; HIDDEN_SIZE=128
SEQ_LEN = N_BITS + DELAY + N_BITS
WORK_DIR = "/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/ai2/asta-test/run_ann_learning_rule/run3/"
N_ITER = 5000
BATCH_SIZE = 32
LOG_INTERVAL = 100

def generate_copy_batch(batch_size, dev=None):
    if dev is None: dev = device
    patterns = torch.randint(0, 2, (batch_size, N_BITS), dtype=torch.float32)
    inputs = torch.zeros(batch_size, SEQ_LEN, INPUT_DIM)
    targets = torch.zeros(batch_size, SEQ_LEN, OUTPUT_DIM)
    output_mask = torch.zeros(batch_size, SEQ_LEN)
    for t in range(N_BITS):
        inputs[:, t, t] = patterns[:, t]
    inputs[:, N_BITS+DELAY-1, N_BITS] = 1.0
    for t in range(N_BITS):
        targets[:, N_BITS+DELAY+t, :] = patterns
    output_mask[:, N_BITS+DELAY:N_BITS+DELAY+N_BITS] = 1.0
    return inputs.to(dev), targets.to(dev), output_mask.to(dev)

def compute_metrics(outputs, targets, output_mask):
    mask = output_mask.unsqueeze(-1)
    mse = ((outputs*mask - targets*mask)**2).sum() / (mask.sum()*OUTPUT_DIM)
    pred_bits = (outputs > 0.5).float()
    correct = ((pred_bits == targets).float() * mask).sum()
    return mse.item(), (correct / (mask.sum()*OUTPUT_DIM)).item()

# ============================================================
# METHOD 1: BPTT
# ============================================================
class VanillaRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(HIDDEN_SIZE, INPUT_DIM)*0.1)
        self.W_rec = nn.Parameter(torch.randn(HIDDEN_SIZE, HIDDEN_SIZE)*0.1)
        self.b_rec = nn.Parameter(torch.zeros(HIDDEN_SIZE))
        self.W_out = nn.Parameter(torch.randn(OUTPUT_DIM, HIDDEN_SIZE)*0.1)
        self.b_out = nn.Parameter(torch.zeros(OUTPUT_DIM))
    def forward(self, inputs):
        B,T,_ = inputs.shape
        h = torch.zeros(B, HIDDEN_SIZE, device=inputs.device)
        outs = []
        for t in range(T):
            h = torch.tanh(h @ self.W_rec.T + inputs[:,t,:] @ self.W_in.T + self.b_rec)
            outs.append(h @ self.W_out.T + self.b_out)
        return torch.stack(outs, dim=1)

def train_bptt(n_iter=N_ITER, lr=0.001):
    torch.manual_seed(SEED)
    model = VanillaRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse_h, acc_h = [], []
    t0 = time.time()
    for it in range(n_iter):
        inp, tgt, msk = generate_copy_batch(BATCH_SIZE)
        opt.zero_grad()
        out = model(inp)
        loss = ((out-tgt)**2 * msk.unsqueeze(-1)).sum() / (msk.sum()*OUTPUT_DIM)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        mse, acc = compute_metrics(out.detach(), tgt, msk)
        mse_h.append(mse); acc_h.append(acc)
        if (it+1) % LOG_INTERVAL == 0:
            print("  BPTT iter", it+1, "| MSE:", round(mse,4), "| Acc:", round(acc,4))
    print("  BPTT time:", round(time.time()-t0, 1), "s")
    return mse_h, acc_h

# ============================================================
# METHOD 4: FA
# ============================================================
def train_fa(n_iter=N_ITER, lr=0.001):
    torch.manual_seed(SEED)
    W_in=(torch.randn(HIDDEN_SIZE,INPUT_DIM)*0.1).to(device)
    W_rec=(torch.randn(HIDDEN_SIZE,HIDDEN_SIZE)*0.1).to(device)
    b_rec=torch.zeros(HIDDEN_SIZE).to(device)
    W_out=(torch.randn(OUTPUT_DIM,HIDDEN_SIZE)*0.1).to(device)
    b_out=torch.zeros(OUTPUT_DIM).to(device)
    B_rec=(torch.randn(HIDDEN_SIZE,HIDDEN_SIZE)*0.1).to(device)
    B_out=(torch.randn(HIDDEN_SIZE,OUTPUT_DIM)*0.1).to(device)
    params=[W_in,W_rec,b_rec,W_out,b_out]
    m=[torch.zeros_like(p) for p in params]
    v=[torch.zeros_like(p) for p in params]
    b1,b2,eps=0.9,0.999,1e-8
    mse_h,acc_h=[],[]
    t0=time.time()
    for it in range(n_iter):
        inp,tgt,msk=generate_copy_batch(BATCH_SIZE)
        h=torch.zeros(BATCH_SIZE,HIDDEN_SIZE,device=device)
        hs,outputs=[],[]
        for t in range(SEQ_LEN):
            h=torch.tanh(h@W_rec.T+inp[:,t,:]@W_in.T+b_rec)
            hs.append(h)
            outputs.append(h@W_out.T+b_out)
        out_t=torch.stack(outputs,dim=1)
        out_err=(out_t-tgt)*msk.unsqueeze(-1)
        dWi=torch.zeros_like(W_in)
        dWr=torch.zeros_like(W_rec)
        dbi=torch.zeros_like(b_rec)
        dWo=torch.zeros_like(W_out)
        dbo=torch.zeros_like(b_out)
        dh_next=torch.zeros(BATCH_SIZE,HIDDEN_SIZE,device=device)
        for t in reversed(range(SEQ_LEN)):
            ht=hs[t]
            hp=hs[t-1] if t>0 else torch.zeros_like(ht)
            xt=inp[:,t,:]
            dt=1-ht**2
            dh=(out_err[:,t,:]@B_out.T+dh_next)*dt
            dWo+=out_err[:,t,:].T@ht/BATCH_SIZE
            dbo+=out_err[:,t,:].mean(0)
            dWr+=dh.T@hp/BATCH_SIZE
            dWi+=dh.T@xt/BATCH_SIZE
            dbi+=dh.mean(0)
            dh_next=dh@B_rec.T
        grads=[dWi,dWr,dbi,dWo,dbo]
        tn=sum(g.norm()**2 for g in grads)**0.5
        if tn>1.0: grads=[g/tn for g in grads]
        ta=it+1
        for i,(pw,g) in enumerate(zip(params,grads)):
            m[i]=b1*m[i]+(1-b1)*g
            v[i]=b2*v[i]+(1-b2)*g**2
            pw.data-=lr*(m[i]/(1-b1**ta))/(((v[i]/(1-b2**ta))**0.5)+eps)
        mse,acc=compute_metrics(out_t.detach(),tgt,msk)
        mse_h.append(mse); acc_h.append(acc)
        if (it+1)%LOG_INTERVAL==0:
            print("  FA iter",it+1,"| MSE:",round(mse,4),"| Acc:",round(acc,4))
    print("  FA time:", round(time.time()-t0,1), "s")
    return mse_h,acc_h

# ============================================================
# METHOD 2&3: PSC
# ============================================================
def train_psc(n_iter=N_ITER, lr=0.001, lr_pred=0.001, beta=0.1, gamma=0.3,
              lam=0.9, T_theta=20, use_oscillatory_gate=True, label="PSC"):
    torch.manual_seed(SEED)
    W_in=(torch.randn(HIDDEN_SIZE,INPUT_DIM)*0.1).to(device)
    W_rec=(torch.randn(HIDDEN_SIZE,HIDDEN_SIZE)*0.1).to(device)
    b_rec=torch.zeros(HIDDEN_SIZE).to(device)
    W_out=(torch.randn(OUTPUT_DIM,HIDDEN_SIZE)*0.1).to(device)
    b_out=torch.zeros(OUTPUT_DIM).to(device)
    W_pred=(torch.randn(HIDDEN_SIZE,HIDDEN_SIZE)*0.01).to(device)
    params=[W_in,W_rec,b_rec,W_out,b_out]
    m_adam=[torch.zeros_like(p) for p in params]
    v_adam=[torch.zeros_like(p) for p in params]
    b1,b2,eps=0.9,0.999,1e-8
    mse_h,acc_h=[],[]
    t0=time.time()
    for it in range(n_iter):
        inp,tgt,msk=generate_copy_batch(BATCH_SIZE)
        h=torch.zeros(BATCH_SIZE,HIDDEN_SIZE,device=device)
        pc=torch.zeros(BATCH_SIZE,HIDDEN_SIZE,device=device)
        er=torch.zeros(BATCH_SIZE,HIDDEN_SIZE,HIDDEN_SIZE,device=device)
        ei=torch.zeros(BATCH_SIZE,HIDDEN_SIZE,INPUT_DIM,device=device)
        dWin=torch.zeros_like(W_in)
        dWr=torch.zeros_like(W_rec)
        dbr=torch.zeros_like(b_rec)
        dWo=torch.zeros_like(W_out)
        dbo=torch.zeros_like(b_out)
        dWp=torch.zeros_like(W_pred)
        outs=[]
        for t in range(SEQ_LEN):
            xt=inp[:,t,:]
            hp=h.clone()
            h=torch.tanh(hp@W_rec.T+xt@W_in.T+b_rec)
            dt=1-h**2
            yt=h@W_out.T+b_out
            outs.append(yt)
            pc=(1-beta)*pc+beta*(hp@W_pred.T)
            dp=h-pc
            er=lam*er+torch.bmm(dt.unsqueeze(2),hp.unsqueeze(1))
            ei=lam*ei+torch.bmm(dt.unsqueeze(2),xt.unsqueeze(1))
            en=er.norm(dim=(1,2),keepdim=True)
            er=torch.where(en>5.0,er*5.0/(en+1e-8),er)
            iso=msk[:,t].bool()
            oe=torch.zeros_like(yt)
            if iso.any(): oe[iso]=yt[iso]-tgt[:,t,:][iso]
            oeh=oe@W_out
            st=gamma*dp+(1-gamma)*oeh
            gate=max(0.0,math.sin(2*math.pi*t/T_theta)) if use_oscillatory_gate else 1.0
            if gate>0:
                dWr+=gate*(st.unsqueeze(2)*er).mean(0)
                dWin+=gate*(st.unsqueeze(2)*ei).mean(0)
                dbr+=gate*st.mean(0)
            dWo+=oe.T@h/BATCH_SIZE
            dbo+=oe.mean(0)
            dWp+=(dp.unsqueeze(2)*hp.unsqueeze(1)).mean(0)
        grads=[dWin,dWr,dbr,dWo,dbo]
        tn=sum(g.norm()**2 for g in grads)**0.5
        if tn>1.0: grads=[g/tn for g in grads]
        ta=it+1
        for i,(pw,g) in enumerate(zip(params,grads)):
            m_adam[i]=b1*m_adam[i]+(1-b1)*g
            v_adam[i]=b2*v_adam[i]+(1-b2)*g**2
            pw.data-=lr*(m_adam[i]/(1-b1**ta))/(((v_adam[i]/(1-b2**ta))**0.5)+eps)
        pn=dWp.norm()
        if pn>1.0: dWp=dWp/pn
        W_pred.data-=lr_pred*dWp
        out_t=torch.stack(outs,dim=1)
        mse,acc=compute_metrics(out_t.detach(),tgt,msk)
        mse_h.append(mse); acc_h.append(acc)
        if (it+1)%LOG_INTERVAL==0:
            print(" ",label,"iter",it+1,"| MSE:",round(mse,4),"| Acc:",round(acc,4))
    print(" ",label,"time:",round(time.time()-t0,1),"s")
    return mse_h,acc_h

# ============================================================
# RUN ALL METHODS
# ============================================================
all_results = {}

print("")
print("=== Running BPTT (5000 iters, lr=0.001) ===")
mse_bptt, acc_bptt = train_bptt(N_ITER, lr=0.001)
all_results["bptt"] = {"mse_history": mse_bptt[::LOG_INTERVAL],
                        "acc_history": acc_bptt[::LOG_INTERVAL],
                        "final_mse": mse_bptt[-1], "final_acc": acc_bptt[-1],
                        "config": {"lr": 0.001, "method": "BPTT"}}

print("")
print("=== Running PSC-osc (5000 iters, best config) ===")
print("  Config: lr=0.001, beta=0.1, gamma=0.3, lam=0.9, T_theta=20")
mse_posc, acc_posc = train_psc(N_ITER, lr=0.001, lr_pred=0.001, beta=0.1,
                                gamma=0.3, lam=0.9, T_theta=20,
                                use_oscillatory_gate=True, label="PSC-osc")
all_results["psc_osc"] = {"mse_history": mse_posc[::LOG_INTERVAL],
                           "acc_history": acc_posc[::LOG_INTERVAL],
                           "final_mse": mse_posc[-1], "final_acc": acc_posc[-1],
                           "config": {"lr":0.001,"beta":0.1,"gamma":0.3,"lam":0.9,"T_theta":20}}

print("")
print("=== Running PSC-nogate (5000 iters, best config) ===")
print("  Config: lr=0.001, beta=0.1, gamma=0.5, lam=0.9")
mse_png, acc_png = train_psc(N_ITER, lr=0.001, lr_pred=0.001, beta=0.1,
                              gamma=0.5, lam=0.9, T_theta=10,
                              use_oscillatory_gate=False, label="PSC-nogate")
all_results["psc_nogate"] = {"mse_history": mse_png[::LOG_INTERVAL],
                              "acc_history": acc_png[::LOG_INTERVAL],
                              "final_mse": mse_png[-1], "final_acc": acc_png[-1],
                              "config": {"lr":0.001,"beta":0.1,"gamma":0.5,"lam":0.9}}

print("")
print("=== Running FA (5000 iters, lr=0.001) ===")
mse_fa, acc_fa = train_fa(N_ITER, lr=0.001)
all_results["fa"] = {"mse_history": mse_fa[::LOG_INTERVAL],
                     "acc_history": acc_fa[::LOG_INTERVAL],
                     "final_mse": mse_fa[-1], "final_acc": acc_fa[-1],
                     "config": {"lr": 0.001, "method": "FA"}}

# Save results
out_path = os.path.join(WORK_DIR, "full_training_results.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)

print("")
print("=== FINAL COMPARISON ===")
for method, res in all_results.items():
    print(" ", method, "| Final MSE:", round(res["final_mse"],4),
          "| Final Acc:", round(res["final_acc"],4))

print("")
print("Results saved to", out_path)
print("=== Step 7: Full Training COMPLETE ===")