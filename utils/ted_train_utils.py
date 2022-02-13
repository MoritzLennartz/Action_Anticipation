import torch 
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_reg_function(real, pred, reg_critirion):
    mask = torch.logical_not(real.eq(0.0))
    loss_ = reg_critirion(real,pred)
    
    mask = mask.float()
    
    loss_ = mask * loss_
    
    return torch.sum(loss_)/torch.sum(mask)



def create_look_ahead_mask(size):
    
    mask = torch.triu(torch.ones((size,size), dtype=torch.float32), diagonal=1)
    return mask


#PADDING MASK
def create_padding_mask(seq):
  #seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  seq = seq.masked_fill(seq != 0, -50)
  seq = seq.masked_fill(seq == 0, 1)
  seq = seq.masked_fill(seq == -50, 0)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  pad_seq = seq[:,:,0].unsqueeze(1)
  pad_seq = pad_seq.unsqueeze(1)
  return pad_seq  # (batch_size, 1, seq_len)


def create_masks(dec_inp):  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(dec_inp.shape[1]).to(device)
  dec_target_padding_mask = create_padding_mask(dec_inp).to(device)
  combined_mask = torch.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return combined_mask.to(device)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    
    return acc

def train(model, train_loader, valid_loader, class_critirion, reg_critirion, cl_lambda, reg_lambda,
          optimizer, checkpoint_filepath, writer, epochs):
    
    best_valid_loss = np.inf
    improvement_ratio = 0.01
    num_steps_wo_improvement = 0
    
    for epoch in range(epochs):
        nb_batches_train = len(train_loader)
        train_acc = 0
        model.train()
        reg_losses = 0.0
        class_losses = 0.0
        full_losses = 0.0

        for x_enc, x_dec, y in train_loader:
            x_enc = x_enc.to(device)
            y = y.reshape(-1,1).to(device)
            
            x_dec_inp = x_dec[:, :-1].to(device)
            x_dec_real = x_dec[:, 1:].to(device)
            
            combined_mask  = create_masks(x_dec_inp).to(device)
            
            out, act = model(x_enc, x_dec_inp, combined_mask)  # ①

            cl_loss = class_critirion(act, y)  # ②
            re_loss = loss_reg_function(out, x_dec_real, reg_critirion)
            f_loss = cl_lambda * cl_loss + reg_lambda * re_loss
            
            model.zero_grad()  # ③

            f_loss.backward()  # ④
            
            full_losses += f_loss.item()
            class_losses += cl_loss.item()
            reg_losses += re_loss.item()

            optimizer.step()  # ⑤
                        
            train_acc += binary_acc(y, torch.round(act))
            
        writer.add_scalar('training Full loss',
            full_losses / nb_batches_train,
            epoch + 1)
        writer.add_scalar('training Class loss',
            class_losses / nb_batches_train,
            epoch + 1) 
        writer.add_scalar('training Reg loss',
            reg_losses / nb_batches_train,
            epoch + 1)        
        writer.add_scalar('training Acc',
            train_acc / nb_batches_train,
            epoch + 1)
            
        print(f"Epoch {epoch}: | Train_Full_Loss {full_losses / nb_batches_train} | Train_Class_Loss {class_losses / nb_batches_train} | Train_Reg_Loss {reg_losses / nb_batches_train} | Train_Acc {train_acc / nb_batches_train} ")
        
        valid_full_loss, valid_class_loss, valid_reg_loss, val_acc = evaluate(model, valid_loader,
                                                                              class_critirion, reg_critirion,
                                                                              cl_lambda, reg_lambda)
        
        writer.add_scalar('validation Full loss',
                          valid_full_loss,
                          epoch + 1)
        writer.add_scalar('validation Class loss',
                          valid_class_loss,
                          epoch + 1) 
        writer.add_scalar('validation Reg loss',
                          valid_reg_loss,
                          epoch + 1) 
        writer.add_scalar('validation Acc',
                          val_acc,
                          epoch + 1)
        
        if (best_valid_loss - valid_class_loss) > np.abs(best_valid_loss * improvement_ratio):
            num_steps_wo_improvement = 0
        else:
            num_steps_wo_improvement += 1
            
        if num_steps_wo_improvement == 7:
            print("Early stopping on epoch:{}".format(str(epoch)))
            break;
        if valid_class_loss <= best_valid_loss:
            best_valid_loss = valid_class_loss  
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'LOSS': full_losses / nb_batches_train,
            }, checkpoint_filepath)


def evaluate(model, data_loader, class_critirion, reg_critirion, cl_lambda, reg_lambda,):
    nb_batches = len(data_loader)
    val_full_losses = 0.0
    val_class_losses = 0.0
    val_reg_losses = 0.0
    with torch.no_grad():
        model.eval()
        acc = 0 
        for x_enc, x_dec, y in data_loader:
            x_enc = x_enc.to(device)
            y = y.reshape(-1,1).to(device)
            
            x_dec_inp = x_dec[:, :-1].to(device)
            x_dec_real = x_dec[:, 1:].to(device)
            
            combined_mask  = create_masks(x_dec_inp).to(device)
            
            out, act = model(x_enc, x_dec_inp, combined_mask)
            
            val_cl_loss = class_critirion(act, y)
            val_re_loss = loss_reg_function(out, x_dec_real, reg_critirion)
            val_f_loss = cl_lambda * val_cl_loss + reg_lambda * val_re_loss
            
            val_full_losses += val_f_loss.item()
            val_class_losses += val_cl_loss.item()
            val_reg_losses += val_re_loss.item()
            
            acc += binary_acc(y, torch.round(act))

    print(f"Valid_Full_Loss {val_full_losses / nb_batches} | Valid_Class_Loss {val_class_losses / nb_batches} | Valid_Reg_Loss {val_reg_losses / nb_batches} | Valid_Acc {acc / nb_batches} \n")
    return val_full_losses / nb_batches, val_class_losses / nb_batches, val_reg_losses / nb_batches, acc / nb_batches


def test(model, data_loader):
    with torch.no_grad():
        model.eval()
        step = 0
        for x_enc, x_dec, y in data_loader:
            x_enc = x_enc.to(device)
            y = y.reshape(-1,1).to(device)
            
            x_dec_inp = x_dec[:, :-1].to(device)
            
            combined_mask  = create_masks(x_dec_inp).to(device)
            
            out, act = model(x_enc, x_dec_inp, combined_mask)
            
            if(step == 0):
                pred = act
                labels = y

            else:
                pred = torch.cat((pred, act), 0)
                labels = torch.cat((labels, y), 0)
            step +=1

    return pred, labels