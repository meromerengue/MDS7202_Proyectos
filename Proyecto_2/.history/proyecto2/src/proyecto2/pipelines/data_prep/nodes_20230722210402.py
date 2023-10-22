"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.11
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.compose import ColumnTransformer

def get_data():
    # TODO: completar get_data()
    df = pd.read_parquet("C:/Users/marti/Desktop/Universidad/IX_Semestre/LPC/Proyecto_2/dataset.pq")
    return df


def preprocess(df):
    df['prop_deuda_sueldo_anual'] = df['outstanding_debt'] / df['annual_income']
    df['prop_emi_sueldo'] = df['total_emi_per_month']/df['monthly_inhand_salary']

    condiciones = df[
        (df['age'] < 0) | (df['age'] > 150) |
        (df['num_bank_accounts'] < 0) |
        (df['num_credit_card'] > 20) |
        (df['num_bank_accounts'] > 20) |
        (df['interest_rate'] > 500) |
        (df['num_of_loan'] > 10) |
        (df['num_of_loan'] < 0) |
        (df['num_of_delayed_payment'].isna()) |
        (df['monthly_inhand_salary'].isna()) |
        (df['num_of_delayed_payment'] < 0) |
        (df['num_of_delayed_payment'] > 30) |
        (df['num_credit_inquiries'].isna()) |
        (df['num_credit_inquiries'] > 20) |
        (df['total_emi_per_month'] > 10000) |
        (df['credit_history_age'].isna()) |
        (df['amount_invested_monthly'].isna()) |
        (df['amount_invested_monthly'] > 2000) |
        (df['monthly_balance'] < 0) |
        (df['payment_behaviour']=='!@9#%8') |
        (df['payment_of_min_amount']=='NM') |
        (df['prop_emi_sueldo'] > 1)
    ]

    df = df.drop(condiciones.index)
    df['monthly_inhand_salary'] = df['monthly_inhand_salary'].fillna(0)
    df['changed_credit_limit'] = df['changed_credit_limit'].fillna(0)
    df['monthly_balance'] = df['monthly_balance'].fillna(0)
    df = df.drop(columns = ['occupation', 'customer_id'])
    def log_transform(x):
        return np.log(x + 2)
    minmax = [
    'age',
    'num_bank_accounts',
    'num_credit_card',
    'num_of_loan',
    'delay_from_due_date',
    'num_of_delayed_payment',
    'changed_credit_limit',
    'num_credit_inquiries',
    'credit_utilization_ratio',
    'credit_history_age'
    ]

    log = [
    'annual_income',
    'monthly_inhand_salary',
    'interest_rate',
    'outstanding_debt',
    'total_emi_per_month',
    'amount_invested_monthly',
    'monthly_balance',
    'prop_deuda_sueldo_anual',
    'prop_emi_sueldo'
    ]
    sino = 'payment_of_min_amount'
    sinord = ['No','Yes']

    onehot = [
    'payment_behaviour'
    ]

    transformer = ColumnTransformer([
    ('min_max_scaler', MinMaxScaler(), minmax),
    ('one_hot_encoder', OneHotEncoder(sparse_output=False), onehot),
    ('caregorical', OrdinalEncoder(categories = sinord), sino),
    ('log', FunctionTransformer(log_transform), log)
    ], remainder='passthrough').set_output(transform='pandas')

    df_final = transformer.fit_transform(df)

    return df_final

