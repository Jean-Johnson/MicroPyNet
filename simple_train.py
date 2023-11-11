from mlp_model import MLP

## sample dataset
xs =[
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0] # true output
model = MLP(3,[4,4,1])


def train():
    print("[!] Training started")
    # Training loop
    epoch = 20
    learning_rate = 0.05
    # model initializion
    for k in range(epoch):
        # do the forwards pass
        ypred = [model(x) for x in xs]
        # calculates loss
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys,ypred))
        # before the backwards pass set the zero grad ie: set all the gradients to zero
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()
        # update parameters: use gradient decent
        for p in model.parameters():
            p.data += -learning_rate * p.grad
        print(f'for epoch {k} : loss is {loss.data}')
    print("[!] Training ended")
    

if __name__ == "__main__":
    print(f"[+] Before Training: ")
    ypred = [model(x) for x in xs]
    print(f"Ground Truth  <-> Predicted")
    for pred, gt in zip(ypred,ys):
        print(f"{gt} <-> {pred.data}")
    train()
    print(f"\n[+] After Training: ")
    ypred = [model(x) for x in xs]
    print(f"Ground Truth  <-> Predicted")
    for pred, gt in zip(ypred,ys):
        print(f"{gt} <-> {pred.data}")
    