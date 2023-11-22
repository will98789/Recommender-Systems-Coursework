import random
import pandas as pd
import json
import numpy as np

print('Loading data...')

ds_b = pd.read_csv('yelp_academic_dataset_business.csv')

ds_b.sort_values(by=['stars'], ascending=False, inplace=True)

businesses = ds_b['business_id'].unique()

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

def show_business(business_id):
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


def business_options(index):
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
    return 'stay'

def business_profile(i, business_id):
    business = ds_b[ds_b['business_id'] == business_id]
    name = business['name'].values[0]
    if len(name) >= 25:
        name = name[:22] + '...'
    catagories = ','.join(business['categories'].values[0].split(',')[:3])
    if len(catagories) >= 40:
        catagories = catagories[:37] + '...'
    rating = business['stars'].values[0]
    return ("{:<3} | {:<25} | {:<40} | {:<6}".format(str(i), name, catagories, str(rating)))

def top_n(n, start):
    business_indexes = ds_b.iloc[start:(n+start)]['business_id'].values
    ratings = ds_b.iloc[start:(n+start)]['stars'].values
    return business_indexes, ratings

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
        show_business(prev_indexes[selection-1-n_ratings*page])
        inp = business_options(prev_indexes[selection-1-n_ratings*page])
    except Exception as e:
        # print(e)
        business_indexes, ratings= top_n(n_ratings, page*n_ratings)
        prev_indexes = business_indexes
        print(f'Top {n_ratings} businesses for you: (Page {page+1})\n')
        print ("{:<3} | {:<25} | {:<40} | {:<6} | {:<4}".format('No.', 'Name', 'Catagories', 'Rating', 'Pred.'))
        print('-'*91)
        for i in range(n_ratings):
            print(business_profile((page*n_ratings+i)+1,business_indexes[i]))
            # print('business_id:',ds_b.loc[ds_b['business_id'] == reverse_business_hash[business_indexes[i]]]['categories'].to_list(),ratings[i])
        print(f'Select a business ({page*n_ratings+1} - {page*n_ratings+n_ratings}), Show more (s), previous page (p), or exit (e)?')
        inp = input()