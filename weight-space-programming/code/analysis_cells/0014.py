import numpy as np, torch, json
F2 = "work/fwd_2"
H = {l: np.load(f"{F2}/hidden_L{l}.npy", mmap_mode="r") for l in range(25)}
TID = np.load(f"{F2}/token_ids.npy"); LP = np.load(f"{F2}/token_logprobs.npy")
TKI = np.load(f"{F2}/topk_ids.npy", mmap_mode="r"); TKL = np.load(f"{F2}/topk_logprobs.npy", mmap_mode="r")
print(H[3].shape, TID.shape, LP.shape, TKI.shape)
# sanity: token ids match the request
req2 = json.load(open("work/req_pass2.json"))
print("token ids match:", all((TID[i, :len(t)] == np.array(t)).all() for i, t in enumerate(req2["token_ids"])))
for i in (0, 20):
    n = P2META[i]["len"]
    for L in (3, 23):
        r_in = torch.from_numpy(H[L][i, :n].astype(np.float32)); r_out = torch.from_numpy(H[L + 1][i, :n].astype(np.float32))
        print("text", i, P2META[i]["kind"], "layer", L, recon.check_layer(ATT[L], r_in, r_out))
