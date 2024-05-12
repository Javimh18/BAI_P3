demo_label_names = {
    0: 'HA4K_120',
    1: 'HB4K_120',
    2: 'HN4K_120',
    3: 'MA4K_120',
    4: 'MB4K_120',
    5: 'MN4K_120'
}

eth_label_names = {
    0: 'Asian',
    1: 'Caucasian',
    2: 'Black',
}

gen_label_names = {
    0: 'Male',
    1: 'Female'
}

def class_mapping(ethnicity:str):
    if 'HA4K_120' in ethnicity:
        return 0
    elif 'HB4K_120' in ethnicity:
        return 1
    elif 'HN4K_120' in ethnicity:
        return 2
    if 'MA4K_120' in ethnicity:
        return 3
    elif 'MB4K_120' in ethnicity:
        return 4
    elif 'MN4K_120' in ethnicity:
        return 5
    else:
        print(f"The label {ethnicity} is not recognized... exiting...")
        exit()