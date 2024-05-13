import time
import torch
import torch.nn as nn
from model.Models import Transformer
# from Process import *
import torch.nn.functional as F
from model.Batch import create_masks
# import dill as pickle
from utils import MyTokenizer, MyMasker
from utils.data import TextDataset
from torch.utils.data import DataLoader, random_split


def train_model(model, epochs=100, printevery=1):
    

    masker = MyMasker()
    tokenizer = MyTokenizer()
    model.train()

    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    '''
    if opt.checkpoint > 0:
        cptime = time.time()
    '''

    print("loading data...")
    dataset = TextDataset()
    train_size = int(0.8*len(dataset))
    test_size = len(dataset)-train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=0)

    print("training model...")
    start = time.time()
    if torch.cuda.is_available():
        print('gpu detected!')
    else:
        print('no gpu detected')
        return 0

    for epoch in range(epochs):

        total_loss = 0

        '''
        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        
        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')
        '''
                    
        for i, trg in enumerate(trainloader):

            # src = batch.src.transpose(0,1)
            # trg = batch.trg.transpose(0,1)
            # trg_input = trg[:, :-1]
            # src_mask, _ = create_masks(src, trg_input) # need to edit

            # src is the incomplete word
            src = masker.mask(trg)  # e.g. [m_zh__n, _s, _w_eso_e]
            src = tokenizer.encode(src)  # e.g. [[], [], []]

            # trg is the complete word
            trg = tokenizer.encode(trg)

            # our src_mask is the same as trg_mask = mask
            mask, _ = create_masks(src)  # e.g. [[1, 1, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0]]

            preds = model(src, mask)
            # ys = trg[:, 1:].contiguous().view(-1)

            optim.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), trg.contiguous().view(-1), ignore_index=0)
            loss.backward()
            optim.step()

            '''
            if opt.SGDR == True: 
                opt.sched.step()
            '''
            total_loss += loss.item()

            
            if (i + 1) % printevery == 0:
                p = int(100 * (i + 1) / train_size)
                avg_loss = total_loss / printevery
                print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss), end='\r')
                '''
                else:
                    print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" %\
                    ((time.time() - start)//60, epoch + 1, "".join('#'*(p//5)), "".join(' '*(20-(p//5))), p, avg_loss))
                '''
                total_loss = 0

            '''
            if ((time.time()-cptime)//60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()
            '''

        #print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" %\
        #((time.time() - start)//60, epoch + 1, "".join('#'*(100//5)), "".join(' '*(20-(100//5))), 100, avg_loss, epoch + 1, avg_loss))

def main():
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', required=True)
    parser.add_argument('-trg_data', required=True)
    parser.add_argument('-src_lang', required=True)
    parser.add_argument('-trg_lang', required=True)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1500)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)

    opt = parser.parse_args()
    
    opt.device = 0 if opt.no_cuda is False else -1
    if opt.device == 0:
        assert torch.cuda.is_available()
    '''

    #read_data(opt)
    #SRC, TRG = create_fields(opt)
    #opt.train = create_dataset(opt, SRC, TRG)
    model = Transformer(src_vocab=28, d_model=512, N=6, heads=8, dropout=0.1)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    '''
    t1 = time.time()9
    word = ['mom']*4
    tokenizer = MyTokenizer()
    masker = MyMasker()

    src = masker.mask(word)
    src = tokenizer.encode(src)
    trg = tokenizer.encode(word)
    mask, _ = create_masks(trg)
    t2 = time.time()
    # x = x.transpose(0, 1)
    # mask_x = mask_x.transpose(0, 1)
    # print(src.shape, mask.shape)
    y = model(src, mask)
    t3 = time.time()
    # print(y.view(-1, y.size(-1)).shape, trg.contiguous().view(-1).shape)
    # loss = F.cross_entropy(y.view(-1, y.size(-1)), trg.contiguous().view(-1), ignore_index=0)
    # print(loss)
    print(model.state_dict())
    
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    if opt.load_weights is not None and opt.floyd is not None:
        os.mkdir('weights')
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    '''
    train_model(model)
    '''
    if opt.floyd is False:
        promptNextAction(model, opt, SRC, TRG)
    '''

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt, SRC, TRG):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = 'weights'

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input("name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res= yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break
            
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            if saved_once == 0:
                pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
                saved_once = 1
            
            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt)
        else:
            print("exiting program...")
            break

    # for asking about further training use while true loop, and return
if __name__ == "__main__":
    main()
