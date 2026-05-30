# Data science 2026-1 Team 1 term project - open source contribution / find best model
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, mean_absolute_error,
    mean_squared_error, r2_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import ( MinMaxScaler, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, RobustScaler, StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore")

RANDOM_STATE = 42 # Global random state

# Fixed preprocessing function
def fixed_preprocess(data, is_train=True, train_stats=None):
    df = data.copy()

    # Replace missing values (GPU)
    cpu_to_gpu_map = {
        'Intel Core i5': 'Intel Iris Xe Graphics',
        'Intel Core i3': 'Intel UHD Graphics',
        'Intel Celeron': 'Intel UHD Graphics 600',
        'Intel Core i7': 'Intel Iris Xe Graphics',
        'AMD Ryzen 5': 'AMD Radeon Graphics',
        'AMD Ryzen 7': 'AMD Radeon Graphics',
        'Intel Pentium': 'Intel UHD Graphics 605',
        'AMD Ryzen 3': 'AMD Radeon Graphics',
        'AMD Athlon': 'AMD Radeon Graphics',
        'Intel Core i9': 'Intel UHD Graphics 770',
        'Apple M1': 'Apple M1 GPU',
        'Apple M2': 'Apple M2 GPU',
        'Apple M2 Pro': 'Apple M2 Pro GPU',
        'Apple M3': 'Apple M3 GPU',
        'Microsoft SQ1': 'Adreno 685',
        'Intel Evo Core i7': 'Intel Iris Xe Graphics',
        'Intel Evo Core i5': 'Intel Iris Xe Graphics',
        'AMD 3020e': 'AMD Radeon Graphics',
        'AMD Radeon 5': 'AMD Radeon Graphics',
        'AMD Radeon 9': 'AMD Radeon Graphics',
        'Qualcomm Snapdragon 7c': 'Adreno 618',
        'Apple M1 Pro': 'Apple M1 Pro GPU',
        'AMD Ryzen 9': 'AMD Radeon Graphics',
        'Intel Core M3': 'Intel HD Graphics 615',
        'Mediatek MT8183': 'Mali-G72 MP3',
        'Qualcomm Snapdragon 7': 'Adreno 618',
        'Qualcomm Snapdragon 8': 'Adreno 690',
        'AMD 3015e': 'AMD Radeon Graphics',
        'AMD 3015Ce': 'AMD Radeon Graphics'
    }
    df['GPU'] = df['GPU'].fillna(df['CPU'].map(cpu_to_gpu_map))

    # Replace missing values (storage, screen)
    if is_train:
        storage_mode = df['Storage type'].mode()[0]
        screen_median = df['Screen'].median()
        train_stats['storage_mode'] = storage_mode
        train_stats['screen_median'] = screen_median
    else:
        storage_mode = train_stats['storage_mode']
        screen_median = train_stats['screen_median']

    df['Storage type'] = df['Storage type'].fillna(storage_mode)
    df['Screen'] = df['Screen'].fillna(screen_median)

    # Basic preprocessing
    df['Touch'] = df['Touch'].map({'Yes': 1, 'No': 0}) # Convert Yes/No to 1/0

    df = df[df['Storage'] > 0] # Remove impossible storage
    df['Storage'] = np.log1p(df['Storage']) #log transformation

    # Remove unusual (mostly wrong) RAM
    valid_ram = [4, 8, 16, 24, 32, 64, 128]
    df = df[df['RAM'].isin(valid_ram)]

    # Merge same brand (but different representation)
    df['Brand'] = df['Brand'].replace('Dynabook Toshiba', 'Toshiba')

    # Remove unusefull features
    df = df.drop(columns=['Laptop', 'Model'], errors='ignore')

    return df

# Function to test variouse encoder
def encode(df, is_train=True, train_stats=None, encoding_method='default'):
    df = df.copy()
    
    # CPU/GPU Score mapping table (based on passmark)
    cpu_scores = {
        'Intel Core i9': 30000, 'Intel Evo Core i9': 30000,
        'Intel Core i7': 25000, 'Intel Evo Core i7': 25000,
        'Intel Core i5': 15000, 'Intel Evo Core i5': 15000,
        'Intel Core i3': 5000,
        'Intel Pentium': 3000, 'Intel Core M3': 2500, 'Intel Celeron': 2000,
        'AMD Ryzen 9': 24000, 'AMD Ryzen 7': 18000, 'AMD Ryzen 5': 10000, 'AMD Ryzen 3': 7000,
        'AMD Athlon': 3000, 'AMD 3020e': 2000, 'AMD 3015e': 1500, 'AMD 3015Ce': 1500,
        'AMD Radeon 9': 24000, 'AMD Radeon 5': 13000,
        'Apple M2 Pro': 16000, 'Apple M1 Pro': 14000, 'Apple M2': 10000, 'Apple M1': 8000,
        'Qualcomm Snapdragon 8': 5000, 'Qualcomm Snapdragon 7': 3500,
        'Microsoft SQ1': 3000, 'Mediatek MT8183': 1500
    }
    gpu_scores = {
        'RTX 4090': 27000, 'RTX 4080': 24000, 'RTX 3080': 16000,
        'RTX 4070': 19000, 'RTX 3070': 15000, 'RTX 4060': 17000,
        'RTX 3060': 13000, 'RTX 4050': 14000, 'RTX 3050': 12000,
        'GTX 1650': 6000, 'MX450': 4000,
        'Apple M2 Pro 19-Core': 12000, 'Apple M1 Pro 16-Core': 10000,
        'Apple M2 10-Core': 6000, 'Apple M1 8-Core': 5000,
        'Radeon RX 6800M': 13000, 'Radeon RX 6600M': 13000,
        'AMD Radeon Graphics': 3000, 'Intel Iris Xe Graphics': 2800,
        'Intel UHD Graphics': 1500, 'Intel UHD Graphics 600': 300
    }

    # encoding method based on domain knowledge (manual encoding method)
    if encoding_method == 'default':
        df['CPU_Score'] = df['CPU'].map(cpu_scores).fillna(2000)
        df['GPU_Score'] = df['GPU'].map(gpu_scores).fillna(1500)
        df = df.drop(['CPU', 'GPU'], axis=1)

        # one hot encoding
        columns = ['Status', 'Brand']

        if is_train:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_result = encoder.fit_transform(df[columns])
            train_stats['onehot_encoder'] = encoder
        else:
            encoder = train_stats['onehot_encoder']
            encoded_result = encoder.transform(df[columns])
        
        encoded_df = pd.DataFrame(encoded_result, 
                                  columns=encoder.get_feature_names_out(columns),
                                  index = df.index
                                  )
        
        df = df.drop(columns=columns)
        df = pd.concat([df, encoded_df], axis=1)

        # Convert boolean to numeric
        for col in df.columns:
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(int)

        # Mapping Storage (SSD > HDD > eMMC)
        storage_tier_map = {
            'eMMC': 0,
            'HDD': 1,
            'SSD': 2
        }
        df['Storage_Tier'] = (df['Storage type'].map(storage_tier_map))

        # Merge unimportant brands (Based on F-Score)
        brand_cols = [c for c in df.columns if c.startswith('Brand_')]

        if is_train:
            target_col = 'Final Price'
            X_brands = df[brand_cols]
            y = df[target_col]

            # f_regression to get score
            f_scores, _ = f_regression(X_brands, y)
            importance_series = pd.Series(f_scores, index=brand_cols)
            frequencies = X_brands.sum()

            FREQ_THRESHOLD = 20 # around 1 % of total
            IMPORTANCE_THRESHOLD = 15

            # Find brands to merge
            brands_to_group = []
            for brand in brand_cols:
                if frequencies[brand] < FREQ_THRESHOLD \
                   and importance_series[brand] < IMPORTANCE_THRESHOLD:
                    brands_to_group.append(brand)

            train_stats['brands_to_group'] = brands_to_group
            print("==========================")
            print(f"list of brand 'others' Total:({len(brands_to_group)})")
            print(brands_to_group)

        # merge brands to brand_others
        target_groups = train_stats.get('brands_to_group', [])
        if target_groups:
            existing_groups \
                = [col for col in target_groups if col in df.columns]

            if existing_groups:
                df['Brand_Others'] = df[existing_groups].sum(axis=1)
                df['Brand_Others'] \
                    = df['Brand_Others'].apply(lambda x: 1 if x > 0 else 0)
                df = df.drop(columns=existing_groups)
            else:
                df['Brand_Others'] = 0
        
        # Remove duplicated columns
        drop_cols = ['Status_Refurbished', 'Storage type']
        df = df.drop([c for c in drop_cols if c in df.columns], axis=1)

    # Onehot encode only
    elif encoding_method == 'onehot':
        columns=['CPU', 'GPU', 'Status', 'Brand', 'Storage type']
        
        if is_train:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_result = encoder.fit_transform(df[columns])
            train_stats['onehot_encoder'] = encoder
        #reuse encoder when encoding testing set
        else:
            encoder = train_stats['onehot_encoder']
            encoded_result = encoder.transform(df[columns])
        
        encoded_df = pd.DataFrame(encoded_result, 
                                  columns=encoder.get_feature_names_out(columns),
                                  index = df.index
                                  )
        
        df = df.drop(columns=columns)
        df = pd.concat([df, encoded_df], axis=1)

        # Convert boolean to numeric (for safety)
        for col in df.columns:
            if df[col].dtype == 'bool':
                df[col] = df[col].astype(int)

        # Remove unused columns
        drop_cols = ['Status_Refurbished', 'Storage type_eMMC']
        df = df.drop([c for c in drop_cols if c in df.columns], axis=1)

    
    # ordinal encoding only
    elif encoding_method == 'ordinal':
        columns=['CPU', 'GPU', 'Status', 'Brand', 'Storage type']
        if is_train:
            encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            df[columns] = encoder.fit_transform(df[columns])
            train_stats['ordinal_encoder'] = encoder
        #reuse encoder when encoding testing set
        else:
            encoder = train_stats['ordinal_encoder']
            df[columns] = encoder.transform(df[columns])

    return df


# Function to test various scaler
def scale(train_pre, test_pre, scaling_method='default'):
    # columns to scale
    num_cols = [
        col for col in ['RAM', 'Storage', 'Screen', 'CPU_Score', 'GPU_Score']
        if col in train_pre.columns
    ]

    # Default choice is Standard
    if scaling_method == 'standard':
        scaler = StandardScaler()
    
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()

    elif scaling_method == 'robust':
        scaler = RobustScaler()

    # transform
    train_pre[num_cols] = scaler.fit_transform(train_pre[num_cols])
    test_pre[num_cols] = scaler.transform(test_pre[num_cols])

    return train_pre, test_pre

# Function to test various regression function
def regression (X_train, y_train_class, y_train_price, X_test, y_test_class, y_test_price, test_pred, regression_method):
    results = []

    # Use real(actual) class to train
    X_train_low = X_train[y_train_class == 0]
    y_train_price_low = y_train_price[y_train_class == 0]

    X_train_high = X_train[y_train_class == 1]
    y_train_price_high = y_train_price[y_train_class == 1]

    X_test_low = X_test[test_pred == 0]
    X_test_high = X_test[test_pred == 1]

    y_test_price_exp = np.expm1(y_test_price)

    # Multiple linear regression method
    if regression_method == 'multiple':
        lr_low = LinearRegression()
        lr_high = LinearRegression()

        lr_low.fit(X_train_low, y_train_price_low)
        lr_high.fit(X_train_high, y_train_price_high)
        
        final_pred = pd.Series(index=X_test.index, dtype=float)

        pred_low = lr_low.predict(X_test_low)
        pred_high = lr_high.predict(X_test_high)

        # Calculate scores
        final_pred.loc[X_test_low.index] = np.expm1(pred_low)
        final_pred.loc[X_test_high.index] = np.expm1(pred_high)

        rmse = np.sqrt(mean_squared_error(y_test_price_exp, final_pred))
        mae = mean_absolute_error(y_test_price_exp, final_pred)
        r2 = r2_score(y_test_price_exp, final_pred)

        # store result
        results.append({
            'regression_method': 'multiple_linear',
            'regression_params': {},
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })

    elif regression_method == 'polynomial':
        # more than degree 3 is hard to compute & impractical
        degrees = [2, 3]

        for degree in degrees:
                # make polynomial features
                poly_low_transformer = PolynomialFeatures(degree=degree, include_bias=False)
                poly_high_transformer = PolynomialFeatures(degree=degree, include_bias=False)

                X_train_low_poly = poly_low_transformer.fit_transform(X_train_low)
                X_train_high_poly = poly_high_transformer.fit_transform(X_train_high)
                X_test_low_poly = poly_low_transformer.transform(X_test_low)
                X_test_high_poly = poly_high_transformer.transform(X_test_high)

                # Polynomial regression is basically linear regression with polynomial features
                poly_low = LinearRegression()
                poly_high = LinearRegression()

                poly_low.fit(X_train_low_poly, y_train_price_low)
                poly_high.fit(X_train_high_poly, y_train_price_high)

                final_pred = pd.Series(index=X_test.index, dtype=float)

                pred_low = poly_low.predict(X_test_low_poly)
                pred_high = poly_high.predict(X_test_high_poly)

                # Calculate scores
                final_pred.loc[X_test_low.index] = np.expm1(pred_low)
                final_pred.loc[X_test_high.index] = np.expm1(pred_high)

                rmse = np.sqrt(mean_squared_error(y_test_price_exp, final_pred))
                mae = mean_absolute_error(y_test_price_exp, final_pred)
                r2 = r2_score(y_test_price_exp, final_pred)

                 # store result
                results.append({
                    'regression_method': 'polynomial',
                    'regression_params': {
                        'degree': degree,
                        },
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                })

    return results

# function to test classification. This function also calls regression function to test the whole process.
def model (X_train, y_train_class, y_train_price, X_test, y_test_class, y_test_price, classification_method, regression_method):
    results = []

    if classification_method == 'decision':
        # Prameters to test
        max_depths = [5, 10, 15, 20]
        min_leafs = [5, 10, 15, 20]
        for depth in max_depths:
            for leaf in min_leafs:
                tree = DecisionTreeClassifier(max_depth=depth, 
                                              min_samples_leaf=leaf, 
                                              random_state=RANDOM_STATE)
                
                tree.fit(X_train, y_train_class)
                test_pred = tree.predict(X_test)

                acc = accuracy_score(y_test_class, test_pred)

                # Calling regression
                reg_results = regression(X_train, y_train_class, y_train_price, 
                                         X_test, y_test_class, y_test_price, 
                                         test_pred, regression_method)

                # Store final results
                for reg_result in reg_results:
                    results.append({
                        'classification_method': 'decision',
                        'classification_params': {
                            'max_depth': depth,
                            'min_samples_leaf': leaf
                        },
                        'classification_accuracy': acc,
                        'regression_method': reg_result['regression_method'],
                        'regression_params': reg_result['regression_params'],
                        'rmse': reg_result['rmse'],
                        'mae': reg_result['mae'],
                        'r2': reg_result['r2']
                    })
    elif classification_method == 'logistic':
        # Prameters to test
        cs = [0.01, 0.1, 1, 10]
        for c in cs:
            # high max_iter to allow model to learn enough
            logistic = LogisticRegression(C=c, max_iter=1000)

            logistic.fit(X_train, y_train_class)
            test_pred = logistic.predict(X_test)
            acc = accuracy_score(y_test_class, test_pred)

            # Calling regression
            reg_results = regression(X_train, y_train_class, y_train_price, 
                                         X_test, y_test_class, y_test_price, 
                                         test_pred, regression_method)
            # Store final results
            for reg_result in reg_results:
                    results.append({
                        'classification_method': 'Logistic',
                        'classification_params': {
                            'C': c,
                        },
                        'classification_accuracy': acc,
                        'regression_method': reg_result['regression_method'],
                        'regression_params': reg_result['regression_params'],
                        'rmse': reg_result['rmse'],
                        'mae': reg_result['mae'],
                        'r2': reg_result['r2']
                    })

    elif classification_method == 'knn':
        # Prameters to test
        ks = [3, 5, 7, 9, 11]

        for k in ks:
            knn = KNeighborsClassifier(n_neighbors=k)

            knn.fit(X_train, y_train_class)
            test_pred = knn.predict(X_test)
            acc = accuracy_score(y_test_class, test_pred)

            # Calling regression
            reg_results = regression(X_train, y_train_class, y_train_price, 
                                         X_test, y_test_class, y_test_price, 
                                         test_pred, regression_method)
            # Store final results
            for reg_result in reg_results:
                    results.append({
                        'classification_method': 'KNN',
                        'classification_params': {
                            'n_neighbors': k,
                        },
                        'classification_accuracy': acc,
                        'regression_method': reg_result['regression_method'],
                        'regression_params': reg_result['regression_params'],
                        'rmse': reg_result['rmse'],
                        'mae': reg_result['mae'],
                        'r2': reg_result['r2']
                    })

    return results

# This function tests for all possible value
# encode - scale - model
def test_combination(train_df, test_df):
    # preprocess
    train_stats_orig = {}

    train_pre = fixed_preprocess(train_df, is_train=True,
                                 train_stats=train_stats_orig)
    test_pre = fixed_preprocess(test_df, is_train=False,
                                 train_stats=train_stats_orig)
    
    # list of possible option
    encoding_methods = ['default', 'onehot', 'ordinal']
    scaling_methods = ['standard', 'minmax', 'robust']
    classification_methods = ['decision', 'logistic', 'knn']
    regression_methods = ['multiple', 'polynomial']
    results = []

    for encoding_method in encoding_methods:
        for scaling_method in scaling_methods:
            train_stats = train_stats_orig.copy()
            # encode and scale
            train_encoded = encode(
                train_pre,
                is_train=True,
                train_stats=train_stats,
                encoding_method=encoding_method
            )

            test_encoded = encode(
                test_pre,
                is_train=False,
                train_stats=train_stats,
                encoding_method=encoding_method
            )

            train_scaled, test_scaled = scale(train_encoded, test_encoded,
                                                scaling_method=scaling_method)
            
            X_train = train_scaled.drop(columns=['Final Price', 'Price Class'])
            y_train_price = train_scaled['Final Price']
            y_train_class = train_scaled['Price Class']

            X_test = test_scaled.drop(columns=['Final Price', 'Price Class'])
            y_test_price = test_scaled['Final Price']
            y_test_class = test_scaled['Price Class']

            # classification and regression
            for classification_method in classification_methods:
                for regression_method in regression_methods:
                    # Skip compute heavy, impractical combination
                    if encoding_method == 'onehot' and regression_method == 'polynomial':
                        print("Skipping...Onehot + poly is too heavy.")
                        continue    
                    # Show current combination
                    print("====================================")
                    print(f"Encoding: {encoding_method}")
                    print(f"Scaling: {scaling_method}")
                    print(f"Classifier: {classification_method}")
                    print(f"Regression: {regression_method}")

                    model_results = model(
                        X_train=X_train,
                        y_train_class=y_train_class,
                        y_train_price=y_train_price,
                        X_test=X_test,
                        y_test_class=y_test_class,
                        y_test_price=y_test_price,
                        classification_method=classification_method,
                        regression_method=regression_method
                    )

                    # store result
                    for result in model_results:
                        result['encoding_method'] = encoding_method
                        result['scaling_method'] = scaling_method
                        results.append(result)

    return pd.DataFrame(results)


#custom K-Fold cross validation
def cross_val_custom(df, n_splits = 10):
    df = df.copy()
    # Fixed pipeline
    df['Final Price'] = np.log1p(df['Final Price']) #use log on price

    # Based on median, classify original data / 0 = low, 1 = high
    median_price = df['Final Price'].median()
    df['Price Class'] = (df['Final Price'] >= median_price).astype(int)
    
    # 10-Fold cross validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True,
                            random_state=RANDOM_STATE
    )

    fold_results = []
    split_result = list(kfold.split(df, df['Price Class']))

    # test for all combinations
    for i, split in enumerate(split_result):
        print("=========================================================")
        print(f"Fold: {i}")
        print("=========================================================")

        train_idx = split[0]
        test_idx = split[1]

        fold_train_df = df.iloc[train_idx].copy()
        fold_test_df = df.iloc[test_idx].copy()

        fold_result_df = test_combination(fold_train_df, fold_test_df)
        fold_results.append(fold_result_df)

    # Gathering result and change to string
    all_results_df = pd.concat(fold_results, ignore_index=True)
    all_results_df['classification_params'] \
        = (all_results_df['classification_params'].astype(str))
    all_results_df['regression_params'] \
        = (all_results_df['regression_params'].astype(str))

    groups = [
        'encoding_method',
        'scaling_method',
        'classification_method',
        'classification_params',
        'regression_method',
        'regression_params'
    ]

    # Calculate mean and std
    cv_result = (
        all_results_df.groupby(groups).agg({
            'classification_accuracy': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'r2': ['mean', 'std']
        }).reset_index()
    )

    cv_result.columns = ['_'.join(col).strip('_') for col in cv_result.columns]
    return cv_result

# Shows top 5 best combination
def show_top5(cv_result):
    metrics = [
        ('rmse_mean', True),
        ('mae_mean', True),
        ('r2_mean', False) # higher is better
    ]

    for metric, asc in metrics:
        print("\n====================================================")
        print(f"TOP 5 based on {metric}")
        print("======================================================")

        # sort and head 5
        top5 = (cv_result.sort_values(by=metric, ascending=asc).head(5))

        cols = [
            'encoding_method',
            'scaling_method',
            'classification_method',
            'classification_params',
            'regression_method',
            'regression_params',
            'classification_accuracy_mean',
            'rmse_mean',
            'mae_mean',
            'r2_mean'
        ]

        print(top5[cols].to_string(index=False))


if __name__ == '__main__':
    # 1. Load data
    df = pd.read_csv('laptops.csv')
    print(f"Initial data shape: {df.shape}")

    # 2. Run custom K-Fold cross validation
    cv_result = cross_val_custom(df, n_splits=10)

    # 3. Show top 5 combinations by RMSE, MAE, and R2
    show_top5(cv_result)
