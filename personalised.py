import random
import pandas as pd
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import json
from platform import system as system_name
import os

device = torch.device('cpu')

print('Loading data...')

ds_b = pd.read_csv('yelp_academic_dataset_business.csv')

ds_bs = pd.read_csv('business_stats.csv')

ds_r = pd.read_csv('filtered_normalised_reviews.csv')

print('Finding users...')

#Get all users and businesses
users = ds_r['user_id'].unique()
user_count = ds_r['user_id'].value_counts().to_dict()
businesses = ds_r['business_id'].unique()


#Create implicit data from reviews
print('Loading implicit data... (this can take some time)')
# print(business_vec.vocabulary_.get('category=Pubs'))

R = torch.zeros([users.shape[0], businesses.shape[0]], dtype=torch.int8)
user_item = np.array(ds_r.groupby(by='user_id')['business_id'], dtype=object)
for i in range(users.shape[0]-1):
    R[i+1][np.array(user_item[i][1])] = 1
    if i % 1000 == 0:
        print(f'User: {i}/{users.shape[0]}', end='\r')
print(f'User: {users.shape[0]}/{users.shape[0]}', end='\r')


# In[ ]:


class YelpDataset(Dataset):

    def __init__(self, user_id, business_id, rating):
        self.user_id = user_id
        self.business_id = business_id
        self.rating = rating

    def __len__(self):
        return len(self.rating)
    
    def __getitem__(self, index):
        return self.user_id[index], self.business_id[index], self.rating[index]
    
class SGD(optim.Optimizer):
    
    def __init__(self, params, names, g=0.001, l=0.005, lseven=0.015):
        defaults = dict(g=g, l=l, lseven=lseven, names=names)
        super(SGD, self).__init__(params, defaults)
        
    def step(self, user, business, prediction, actual, rated_by_user, closure=None):
        group = self.param_groups[0]
        pred_error = torch.nan_to_num(torch.sub(actual, prediction))
        implicit_businesses = torch.nonzero(rated_by_user)
        implicit_index = implicit_businesses[:,1]
#         print(self.param_groups)
        l = group['l']
        lseven = group['lseven']
        g = group['g']
        param_names = group['names']
        qi_p = None
        pu_p = None
        y_p = None
        for i, param in enumerate(group['params']):
#                 print(param)
            name = param_names[i]
            if name == 'bu.weight':
                param.data.index_add_(0, user, (pred_error-l*param.data[user]).sum(1).view(-1,1),alpha=g)
            elif name == 'bi.weight':
                param.data.index_add_(0, business, (pred_error-l*param.data[business]).sum(1).view(-1,1),alpha=g)
            elif name == 'qi.weight':
                qi_p = param
            elif name == 'pu.weight':
                pu_p = param
            elif name == 'y.weight':
                y_p = param
        mag_R = 1/torch.sqrt(rated_by_user.sum(1, keepdim=True))
        plus = mag_R * rated_by_user.matmul(y_p.data)
#         qi_a = ((pu_p.data[user] + plus)*(pred_error[:, None])-lseven*qi_p.data[business])
        qi_a = ((pu_p.data[user])*(pred_error[:, None])-lseven*qi_p.data[business])
        pu_a = (qi_p.data[business]*(pred_error[:, None])-lseven*pu_p.data[user])
#             print(pred_error[implicit_businesses[:,0],None].shape, mag_R[implicit_businesses[:,0]].shape, qi_p.data[implicit_index].shape, (lseven*y_p.data[implicit_index]).shape)
        y_a = pred_error[implicit_businesses[:,0], None] * mag_R[implicit_businesses[:,0]] * qi_p.data[implicit_index] - lseven*y_p.data[implicit_index]
        y_p.data.index_add_(0,implicit_index, y_a, alpha=g)
        qi_p.data.index_add_(0, business, qi_a,alpha=g)
        pu_p.data.index_add_(0, user, pu_a,alpha=g)
        return pred_error
        
class SVD(nn.Module):

    def __init__(self, n_users, n_businesses, mean_rating, factors):
        super(SVD, self).__init__()
        self.mean = nn.Parameter(torch.FloatTensor([mean_rating]), False)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_businesses, 1)
        self.qi = nn.Embedding(n_businesses, factors)
        self.pu = nn.Embedding(n_users, factors)
        self.y = nn.Embedding(n_businesses, factors)
        self.qi.weight.data.normal_(0, 0.1)
        self.pu.weight.data.normal_(0, 0.1)
        self.bu.weight.data.uniform_(-0.01, 0.01)
        self.bi.weight.data.uniform_(-0.01, 0.01)
        self.y.weight.data.normal_(0, 0.1)
        self.factors = factors
        self.n_businesses = n_businesses


    def forward(self, user, item, R):
        u_bias = self.bu(user).squeeze()
        i_bias = self.bi(item).squeeze()
        y = self.y(torch.arange(self.n_businesses).to(device))
        R_mult = torch.sqrt(R.sum(1, keepdim=True))
#         x = N.matmul(y) / N_mult
        x = torch.nan_to_num(R.matmul(y) / R_mult)
        predict = ((self.pu(user) + x) * self.qi(item)).sum(1) + u_bias + i_bias + self.mean
        return predict
    

gamma = 0.0034
lambda4 = 0.002

k_factors = 200

print('Creating model...')

params = torch.load('save.chkpt', map_location=device)

if 'users' not in params:
    num_users = len(users)
else:
    num_users = params['users']

model = SVD(num_users, businesses.shape[0], ds_r['stars'].mean(), k_factors).to(device)

# optimiser = SGD(model.parameters(), [name for name, _ in model.named_parameters()], g=gamma, l=lambda4)
optimiser = optim.SGD(model.parameters(), gamma, weight_decay=lambda4)

loss_func = nn.L1Loss().to(device)

print('Loading model parameters...')

# m = torch.zeros([users.shape[0], businesses.shape[0]])
# ui = np.array(x_train.groupby('user_id'))
model.load_state_dict(params['model'])
optimiser.load_state_dict(params['optimizer'])

def filter_attributes(attributes):
    attributes = attributes.replace("{'", '{"')
    attributes = attributes.replace("'}",'"}')
    attributes = attributes.replace("':",'":')
    attributes = attributes.replace(": '", ': "')
    attributes = attributes.replace("',", '",')
    attributes = attributes.replace(", '",', "')
    attributes = attributes.replace("'{", '{')
    attributes = attributes.replace('}"', '}')
    attributes = attributes.replace('"{','{')
    attributes = attributes.replace('}"','}')
    attributes = attributes.replace('None', 'null')
    attributes = attributes.replace('"True"', 'True')
    attributes = attributes.replace('"False"', 'False')
    attributes = attributes.replace('False','"False"')
    attributes = attributes.replace('True','"True"')
    return attributes

def show_business(business_id, user):
    business = ds_b[ds_b['business_id'] == business_id]
    name = business['name'].values[0]
    catagories = business['categories'].values[0].split(',')
    catagory_strings = ['', '']
    current_string = 0
    for i in range(len(catagories)):
        if len(catagory_strings[current_string]) + len(catagories[i]) >= 53:
            current_string += 1
            if current_string > 1:
                catagory_strings += ['']
        catagory_strings[current_string] += catagories[i] + ', '
    catagory_strings[-1] = catagory_strings[-1][:-2]
    rating = business['stars'].values[0]
    print('\n')
    print('{:<40} | {:<10} | {:<35}'.format(name, str(rating)+' \u2606', business['address'].values[0]))
    print('{:<53} | {:<35}'.format('-'*53, business['city'].values[0]))
    print('{:<53} | {:<35}'.format(catagory_strings[0], business['state'].values[0]))
    print('{:<53} | {:<35}'.format(catagory_strings[1], business['postal_code'].values[0]))
    for string in catagory_strings[2:]:
        print('{:<53} | {:<35}'.format(string, ''))
    attributes_string = ''
    # print(business['attributes'].values[0])
    attributes = business['attributes'].values[0]
    attributes = filter_attributes(attributes)
    try:
        attributes = json.loads(attributes)
    except:
        return
    print('-'*(53+35+3))
    to_delete = []
    to_add = {}
    for attribute in attributes:
        if type(attributes[attribute]) == dict:
            for sub_attribute in attributes[attribute]:
                to_add[attribute + ' - ' + sub_attribute] = attributes[attribute][sub_attribute]
            to_delete += [attribute]
    for attribute in to_delete:
        del attributes[attribute]
    for attribute in to_add:
        attributes[attribute] = to_add[attribute]
    for i, attribute in enumerate(attributes):
        if i % 2 == 1:
            attributes_string += ' | '
        attributes_string += '{:<37} {:<6}'.format(attribute[:37], attributes[attribute][:6])
        if i % 2 == 1:
            attributes_string += '\n'
        if i >= 10:
            break
    print(attributes_string)
    return

def rate_business(user, business_id, rating):
    global ds_r
    global R
    # print(business_id)
    stats = ds_bs[ds_bs['business_id'] == business_id]
    # print(stats)
    rating = (rating-stats['mean'])/stats['std']
    rating = rating.to_numpy()[0]
    # print(rating.values()[0])
    model.train()
    loss_val = 5
    user = torch.tensor([user]).to(device)
    business = torch.tensor([business_id]).to(device)
    rating = torch.tensor([rating], dtype=torch.float).to(device)
    count = 0
    while loss_val > 1 and count < 10:
        optimiser.zero_grad()
        ##Train model on index with rating
        prediction = model(user, business, R[user].type(torch.float))
        # print(prediction)
        loss = loss_func(prediction, rating)
        loss_val = loss.item()
        # print(loss, end='\n')
        loss.backward()
        optimiser.step()
        # optimiser.step(user, business, prediction, rating, R[user].type(torch.float))
        count += 1
    model.eval()
    ds_r = pd.concat([ds_r, pd.DataFrame({'user_id': [user.item()], 'business_id': [business_id], 'stars': [rating.item()]})])
    # R[user][business_id] = 1


def business_options(user, index):
    print('How would you rate this business? (1 - 5) or (b) to return')
    inp = ''
    rating = 0
    while inp == '':
        inp = input()
        if inp == 'b':
            return 'stay'
        try:
            rating = int(inp)
            if rating < 1:
                inp = ''
            if rating > 5:
                inp = ''
        except:
            inp = ''
    rate_business(user, index, rating)
    return 'stay'

def business_profile(i, business_id, prediction):
    business = ds_b[ds_b['business_id'] == business_id]
    name = business['name'].values[0]
    if len(name) >= 25:
        name = name[:22] + '...'
    catagories = ','.join(business['categories'].values[0].split(',')[:3])
    if len(catagories) >= 40:
        catagories = catagories[:37] + '...'
    rating = business['stars'].values[0]
    stats = ds_bs[ds_bs['business_id'] == business_id]
    prediction = min(5.0,round((prediction*stats['std'].values[0]+stats['mean'].values[0]), 1))
    return ("{:<3} | {:<25} | {:<40} | {:<6} | {:<5}".format(str(i), name, catagories, str(rating), prediction))

def top_n(predictions, n=10, start_index=0):
    predictions = predictions.cpu().detach().numpy()
    top_n = np.argsort(predictions)[::-1][start_index:(n+start_index)]
    return top_n, predictions[top_n]

def predictions(user_id, model, new_user = False):
    global businesses
    user = torch.LongTensor([user_id]).to(device)
    u_r = ds_r[ds_r['user_id'] == user_id]
    viewed_businesses = u_r['business_id'].unique().tolist()
    ##Get business ids that user has not viewed
    not_viewed = [b for b in businesses if b not in viewed_businesses]
    # print(not_viewed)
    business_in = torch.LongTensor(not_viewed).to(device)
    predictions = model(user, business_in, R[user].type(dtype=torch.float))
    return predictions


def create_user(user_id):
    global R
    print('Generating new user profile...')
    old_bu = model.bu.weight.data
    old_pu = model.pu.weight.data

    model.bu = nn.Embedding(users.shape[0] + 1, 1).to(device)
    model.pu = nn.Embedding(users.shape[0] + 1, k_factors).to(device)

    model.bu.weight.data[:old_bu.shape[0]] = old_bu
    model.pu.weight.data[:old_pu.shape[0]] = old_pu

    # for group in optimiser.param_groups:
    #     param_names = group['names']
    #     for i, param in enumerate(group['params']):
    #         if param_names[i] == 'bu.weight' or param_names[i] == 'pu.weight':
    #             ##Add row of zeros to the end of the matrix
    #             param.data = torch.cat((param.data, torch.zeros(1, param.data.shape[1]).to(device)), 0)
    
    new_implicit = torch.zeros((1, businesses.shape[0]))
    R = torch.concat([R, new_implicit])

print('\nKaggle Dataset Recommender System')
print('Type "exit" to exit\n')

user_id = False

print(f'Please enter your user id (0 - {model.bu.num_embeddings-2}) (or type "-1" to be entered as a new user):')

while user_id == False:
    inp = input()
    try:
        if inp == "e":
            exit()
        if inp == "0":
            user_id = 0 
        user_id = int(inp)
        if user_id < -1 or user_id >= model.bu.num_embeddings-1:
            user_id = False
            print('Invalid user id')
            continue
    except:
        user_id = False
        print('Invalid user id')
        continue

if user_id == -1:
    user_id = model.bu.num_embeddings - 1
    create_user(user_id)

model.eval()

if system_name().lower().startswith('win'):
        os.system('cls')
else:
    os.system('clear')

n_ratings = 8

inp = ''

page = -1

prev_indexes = []

while inp != 'e':
    if inp == 's' or inp == '':
        page += 1
    elif inp == 'stay':
        page = page
    elif inp == 'p':
        page = max(0, page - 1)
    try:
        selection = int(inp)
        if selection-1-n_ratings*page < 0 or selection-1-n_ratings*page >= n_ratings:
            raise Exception
        show_business(businesses[prev_indexes[selection-1-n_ratings*page]], user_id)
        inp = business_options(user_id, prev_indexes[selection-1-n_ratings*page])
    except Exception as e:
        # print(e)
        business_indexes, ratings= top_n(predictions(user_id, model), n_ratings, page*n_ratings)
        prev_indexes = business_indexes
        print(f'Top {n_ratings} businesses for you (user {user_id}): (Page {page+1})\n')
        print ("{:<3} | {:<25} | {:<40} | {:<6} | {:<4}".format('No.', 'Name', 'Catagories', 'Rating', 'Pred.'))
        print('-'*91)
        for i in range(n_ratings):
            print(business_profile((page*n_ratings+i)+1,businesses[business_indexes[i]],ratings[i]))
            # print('business_id:',ds_b.loc[ds_b['business_id'] == reverse_business_hash[business_indexes[i]]]['categories'].to_list(),ratings[i])
        print(f'Select a business ({page*n_ratings+1} - {page*n_ratings+n_ratings}), Show more (s), previous page (p), or exit (e)?')
        inp = input()
    if system_name().lower().startswith('win'):
        os.system('cls')
    else:
        os.system('clear')

print('Saving model...')
torch.save({'model': model.state_dict(),
            'optimizer': optimiser.state_dict(),
            'users': model.bu.num_embeddings}, 'save.chkpt')

print('Saving reviews...')
ds_r.to_csv('filtered_normalised_reviews.csv', index=False)