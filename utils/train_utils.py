import torch 
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    
    return acc


def train(model, train_loader, valid_loader, critirion, optimizer, checkpoint_filepath, writer, epochs):
    
    best_valid_loss = np.inf
    improvement_ratio = 0.01
    num_steps_wo_improvement = 0
    
    for epoch in range(epochs):
        nb_batches_train = len(train_loader)
        train_acc = 0
        model.train()
        losses = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.reshape(-1,1).to(device)
            
            out = model(x)  # ①

            loss = critirion(out, y)  # ②
            
            model.zero_grad()  # ③

            loss.backward()  # ④
            losses += loss.item()

            optimizer.step()  # ⑤
                        
            train_acc += binary_acc(y, torch.round(out))
            
        writer.add_scalar('training loss',
            losses / nb_batches_train,
            epoch + 1)
        writer.add_scalar('training Acc',
            train_acc / nb_batches_train,
            epoch + 1)
            
        print(f"Epoch {epoch}: | Train_Loss {losses / nb_batches_train} | Train_Acc {train_acc / nb_batches_train} ")
        valid_loss, val_acc = evaluate(model, valid_loader, critirion)
        writer.add_scalar('validation loss',
                          valid_loss,
                          epoch + 1)
        writer.add_scalar('validation Acc',
                          val_acc,
                          epoch + 1)
        
        if (best_valid_loss - valid_loss) > np.abs(best_valid_loss * improvement_ratio):
            num_steps_wo_improvement = 0
        else:
            num_steps_wo_improvement += 1
            
        if num_steps_wo_improvement == 10:
            print("Early stopping on epoch:{}".format(str(epoch)))
            break;
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss  
            print("Save best model on epoch:{}".format(str(epoch)))
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'LOSS': losses / nb_batches_train,
            }, checkpoint_filepath)


def evaluate(model, data_loader, critirion):
    nb_batches = len(data_loader)
    val_losses = 0.0
    with torch.no_grad():
        model.eval()
        acc = 0 
        for x, y in data_loader:
            x = x.to(device)
            y = y.reshape(-1,1).to(device)
                    
            out = model(x) 
            val_loss = critirion(out, y)
            val_losses += val_loss.item()
            
            acc += binary_acc(y, torch.round(out))

    print(f"Validation_Loss {val_losses / nb_batches} | Val_Acc {acc / nb_batches} \n")
    return val_losses / nb_batches, acc / nb_batches
    
    
def test(model, data_loader):
    with torch.no_grad():
        model.eval()
        step = 0
        for x, y in data_loader:
            x = x.to(device)
            y = y.reshape(-1,1).to(device)
                    
            out = model(x)    
            if(step == 0):
                pred = out
                labels = y

            else:
                pred = torch.cat((pred, out), 0)
                labels = torch.cat((labels, y), 0)
            step +=1

    return pred, labels