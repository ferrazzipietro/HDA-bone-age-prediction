import os


def data_append(X, age, sex, img_name, age_of_this, sex_of_this, full_path, label):
    img_name = os.path.join(full_path, label, 'cut', img_name)
    X.append(img_name)
    age.append(age_of_this)
    sex.append(sex_of_this)
    return (X, age, sex)

def load_imagesPath_ages_sex(full_path, train_truth, validation_truth, test_truth, only_some_data):
    X_train = []
    X_validation = []
    X_test = []
    age_train = []
    age_validation = []
    age_test = []
    sex_train = []
    sex_validation = []
    sex_test = []
    labels = ['train','validation', 'test']

    for l_idx, label in enumerate(labels):
        path = os.path.join(full_path, label, 'cut')
        image_names = os.listdir(path) 
        print('prendo le image_names da ', path)
        for i, image_name in enumerate(image_names[:only_some_data]):
            
            if not image_name.endswith('.png'):
                continue

            id = int(image_name[:-4])

            if label == 'train':
                if not (id in train_truth['id'].to_list()):
                    continue
                idx = (train_truth['id'] == id)
                age_of_this = int(train_truth['boneage'][idx])
                sex_of_this = train_truth['male'][idx].bool()
                X_train, age_train, sex_train = data_append(X_train, age_train, sex_train, image_name, age_of_this, sex_of_this, full_path, label)
            
            if label == 'validation':
                if not (id in validation_truth['Image ID'].to_list()):
                    continue
                idx = (validation_truth['Image ID'] == id)
                age_of_this = int(validation_truth['Bone Age (months)'][idx])
                sex_of_this = validation_truth['male'][idx].bool()
                X_validation, age_validation, sex_validation = data_append(X_validation, age_validation, sex_validation, image_name, age_of_this, sex_of_this, full_path, label)
        
            if label == 'test':
                if not (id in test_truth['Case ID'].to_list()):
                    continue
                idx = (test_truth['Case ID'] == id)
                age_of_this = int(test_truth['Ground truth bone age (months)'][idx])
                sex_of_this = (test_truth['Sex'][idx]).values[0]
                if sex_of_this=='M':
                    sex_of_this = True
                if  sex_of_this=='F':
                    sex_of_this = False
                X_test, age_test, sex_test = data_append(X_test, age_test, sex_test, image_name, age_of_this, sex_of_this, full_path, label)

    return(X_train, age_train, sex_train, X_validation, age_validation, sex_validation, X_test, age_test, sex_test)