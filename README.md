
# Analysis on Clinical and Financial Data of Patients with a Certain Condition

#### Matthew Zakharia Hadimaja

Let's start by importing some libraries


```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
```

Next, we load the .csv files.


```python
bill_amount = pd.read_csv('bill_amount.csv')
bill_id = pd.read_csv('bill_id.csv')
clinical_data = pd.read_csv('clinical_data.csv')
demographics = pd.read_csv('demographics.csv')
```

## Data Cleaning


```python
import datetime as dt
from datetime import datetime
from dateutil.parser import parse
```

### `bill_amount`


```python
bill_amount.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bill_id</th>
      <th>amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40315104</td>
      <td>1552.634830</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2660045161</td>
      <td>1032.011951</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1148334643</td>
      <td>6469.605351</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3818426276</td>
      <td>755.965425</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9833541918</td>
      <td>897.347816</td>
    </tr>
  </tbody>
</table>
</div>




```python
bill_amount.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13600 entries, 0 to 13599
    Data columns (total 2 columns):
    bill_id    13600 non-null int64
    amount     13600 non-null float64
    dtypes: float64(1), int64(1)
    memory usage: 212.6 KB
    

As can be seen, we have no missing data in this table. The data are also stored in appropriate formats.

### `bill_id`


```python
bill_id.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bill_id</th>
      <th>patient_id</th>
      <th>date_of_admission</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7968360812</td>
      <td>1d21f2be18683991eb93d182d6b2d220</td>
      <td>2011-01-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6180579974</td>
      <td>62bdca0b95d97e99e1c712048fb9fd09</td>
      <td>2011-01-01</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7512568183</td>
      <td>1d21f2be18683991eb93d182d6b2d220</td>
      <td>2011-01-01</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3762633379</td>
      <td>62bdca0b95d97e99e1c712048fb9fd09</td>
      <td>2011-01-01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7654730355</td>
      <td>1d21f2be18683991eb93d182d6b2d220</td>
      <td>2011-01-01</td>
    </tr>
  </tbody>
</table>
</div>




```python
bill_id.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13600 entries, 0 to 13599
    Data columns (total 3 columns):
    bill_id              13600 non-null int64
    patient_id           13600 non-null object
    date_of_admission    13600 non-null object
    dtypes: int64(1), object(2)
    memory usage: 318.8+ KB
    

We have no missing data here. However, `date_of_admission` is not formatted correctly. We will convert the column into its appropriate `datetime` format.


```python
bill_id['date_of_admission'] = pd.to_datetime(bill_id['date_of_admission'])
```


```python
bill_id.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 13600 entries, 0 to 13599
    Data columns (total 3 columns):
    bill_id              13600 non-null int64
    patient_id           13600 non-null object
    date_of_admission    13600 non-null datetime64[ns]
    dtypes: datetime64[ns](1), int64(1), object(1)
    memory usage: 318.8+ KB
    

### `demographics`


```python
demographics.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>gender</th>
      <th>race</th>
      <th>resident_status</th>
      <th>date_of_birth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>fa2d818b2261e44e30628ad1ac9cc72c</td>
      <td>Female</td>
      <td>Indian</td>
      <td>Singaporean</td>
      <td>1971-05-14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5b6477c5de78d0b138e3b0c18e21d0ae</td>
      <td>f</td>
      <td>Chinese</td>
      <td>Singapore citizen</td>
      <td>1976-02-18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>320aa16c61937447fd6631bf635e7fde</td>
      <td>Male</td>
      <td>Chinese</td>
      <td>Singapore citizen</td>
      <td>1982-07-03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>c7f3881684045e6c49020481020fae36</td>
      <td>Male</td>
      <td>Malay</td>
      <td>Singapore citizen</td>
      <td>1947-06-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>541ad077cb4a0e64cc422673afe28aef</td>
      <td>m</td>
      <td>Chinese</td>
      <td>Singaporean</td>
      <td>1970-12-12</td>
    </tr>
  </tbody>
</table>
</div>




```python
demographics.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3000 entries, 0 to 2999
    Data columns (total 5 columns):
    patient_id         3000 non-null object
    gender             3000 non-null object
    race               3000 non-null object
    resident_status    3000 non-null object
    date_of_birth      3000 non-null object
    dtypes: object(5)
    memory usage: 117.3+ KB
    

No missing data here, but the supposedly categorical variables `gender`, `race`, and `resident_status` are not consistent. We also need to convert `date_of_birth` into `datetime` format.

We start by listing down the unique values in each of the categorical variables.


```python
print(demographics['gender'].unique())
print(demographics['race'].unique())
print(demographics['resident_status'].unique())
```

    ['Female' 'f' 'Male' 'm']
    ['Indian' 'Chinese' 'Malay' 'chinese' 'India' 'Others']
    ['Singaporean' 'Singapore citizen' 'PR' 'Foreigner']
    

Then, we replace the redundant values.


```python
clean_demographics = {'gender': {'m': 'Male',
                                 'f': 'Female'},
                      'race': {'chinese': 'Chinese',
                               'India': 'Indian'},
                      'resident_status': {'Singapore citizen': 'Singaporean'}}
demographics.replace(clean_demographics, inplace=True)
```

We also convert `date_of_birth` into `datetime` format.


```python
demographics['date_of_birth'] = pd.to_datetime(demographics['date_of_birth'])
```


```python
demographics.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3000 entries, 0 to 2999
    Data columns (total 5 columns):
    patient_id         3000 non-null object
    gender             3000 non-null object
    race               3000 non-null object
    resident_status    3000 non-null object
    date_of_birth      3000 non-null datetime64[ns]
    dtypes: datetime64[ns](1), object(4)
    memory usage: 117.3+ KB
    


```python
print(demographics['gender'].unique())
print(demographics['race'].unique())
print(demographics['resident_status'].unique())
```

    ['Female' 'Male']
    ['Indian' 'Chinese' 'Malay' 'Others']
    ['Singaporean' 'PR' 'Foreigner']
    

As we can see, the data are in correct format, with the categorical variables grouped accordingly.

### `clinical_data`


```python
clinical_data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date_of_admission</th>
      <th>date_of_discharge</th>
      <th>medical_history_1</th>
      <th>medical_history_2</th>
      <th>medical_history_3</th>
      <th>medical_history_4</th>
      <th>medical_history_5</th>
      <th>medical_history_6</th>
      <th>medical_history_7</th>
      <th>...</th>
      <th>symptom_1</th>
      <th>symptom_2</th>
      <th>symptom_3</th>
      <th>symptom_4</th>
      <th>symptom_5</th>
      <th>lab_result_1</th>
      <th>lab_result_2</th>
      <th>lab_result_3</th>
      <th>weight</th>
      <th>height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1d21f2be18683991eb93d182d6b2d220</td>
      <td>2011-01-01</td>
      <td>2011-01-11</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13.2</td>
      <td>30.9</td>
      <td>123.0</td>
      <td>71.3</td>
      <td>161.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>62bdca0b95d97e99e1c712048fb9fd09</td>
      <td>2011-01-01</td>
      <td>2011-01-11</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>13.8</td>
      <td>22.6</td>
      <td>89.0</td>
      <td>78.4</td>
      <td>160.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c85cf97bc6307ded0dd4fef8bad2fa09</td>
      <td>2011-01-02</td>
      <td>2011-01-13</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>11.2</td>
      <td>26.2</td>
      <td>100.0</td>
      <td>72.0</td>
      <td>151.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>e0397dd72caf4552c5babebd3d61736c</td>
      <td>2011-01-02</td>
      <td>2011-01-14</td>
      <td>0</td>
      <td>1.0</td>
      <td>No</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>13.3</td>
      <td>28.4</td>
      <td>76.0</td>
      <td>64.4</td>
      <td>152.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>94ade3cd5f66f4584902554dff170a29</td>
      <td>2011-01-08</td>
      <td>2011-01-16</td>
      <td>0</td>
      <td>0.0</td>
      <td>No</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>12.0</td>
      <td>27.8</td>
      <td>87.0</td>
      <td>55.6</td>
      <td>160.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
clinical_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3400 entries, 0 to 3399
    Data columns (total 26 columns):
    id                    3400 non-null object
    date_of_admission     3400 non-null object
    date_of_discharge     3400 non-null object
    medical_history_1     3400 non-null int64
    medical_history_2     3167 non-null float64
    medical_history_3     3400 non-null object
    medical_history_4     3400 non-null int64
    medical_history_5     3096 non-null float64
    medical_history_6     3400 non-null int64
    medical_history_7     3400 non-null int64
    preop_medication_1    3400 non-null int64
    preop_medication_2    3400 non-null int64
    preop_medication_3    3400 non-null int64
    preop_medication_4    3400 non-null int64
    preop_medication_5    3400 non-null int64
    preop_medication_6    3400 non-null int64
    symptom_1             3400 non-null int64
    symptom_2             3400 non-null int64
    symptom_3             3400 non-null int64
    symptom_4             3400 non-null int64
    symptom_5             3400 non-null int64
    lab_result_1          3400 non-null float64
    lab_result_2          3400 non-null float64
    lab_result_3          3400 non-null float64
    weight                3400 non-null float64
    height                3400 non-null float64
    dtypes: float64(7), int64(15), object(4)
    memory usage: 690.7+ KB
    

This table contains clinical information of each patient at each admission. In addition to the patient's id, weight, and height, we have 7 medical history data, 6 pre-op medication data, 5 symptom data, and 3 lab result data. Information on the discharge date is also provided in the table.

In this table, we have 2 variables that have missing data, `medical_history_2` and `medical_history_5`. We can also see that `medical_history_3` is not in the correct format, we will later check it. Also, we need to fix the dates again.


```python
clinical_data['date_of_admission'] = pd.to_datetime(clinical_data['date_of_admission'])
clinical_data['date_of_discharge'] = pd.to_datetime(clinical_data['date_of_discharge'])
```

As said previously, we will check `medical_history_3`.


```python
clinical_data['medical_history_3'].unique()
```




    array(['0', 'No', '1', 'Yes'], dtype=object)



We need to change the `No` and `Yes` into `0` and `1` respectively. However, note that the existing `0` and `1` are string, not integer. We will also replace them.


```python
clean_clinical = {'medical_history_3': {'0': 0,
                                        'No': 0,
                                        '1': 1,
                                        'Yes': 1}}
clinical_data.replace(clean_clinical, inplace=True)
```


```python
clinical_data['medical_history_3'].unique()
```




    array([0, 1], dtype=int64)




```python
clinical_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3400 entries, 0 to 3399
    Data columns (total 26 columns):
    id                    3400 non-null object
    date_of_admission     3400 non-null datetime64[ns]
    date_of_discharge     3400 non-null datetime64[ns]
    medical_history_1     3400 non-null int64
    medical_history_2     3167 non-null float64
    medical_history_3     3400 non-null int64
    medical_history_4     3400 non-null int64
    medical_history_5     3096 non-null float64
    medical_history_6     3400 non-null int64
    medical_history_7     3400 non-null int64
    preop_medication_1    3400 non-null int64
    preop_medication_2    3400 non-null int64
    preop_medication_3    3400 non-null int64
    preop_medication_4    3400 non-null int64
    preop_medication_5    3400 non-null int64
    preop_medication_6    3400 non-null int64
    symptom_1             3400 non-null int64
    symptom_2             3400 non-null int64
    symptom_3             3400 non-null int64
    symptom_4             3400 non-null int64
    symptom_5             3400 non-null int64
    lab_result_1          3400 non-null float64
    lab_result_2          3400 non-null float64
    lab_result_3          3400 non-null float64
    weight                3400 non-null float64
    height                3400 non-null float64
    dtypes: datetime64[ns](2), float64(7), int64(16), object(1)
    memory usage: 690.7+ KB
    

The table is cleaned now. Note that `medical_history_3` and `medical_history_5` are in float format because of the missing values. Since we don't know cause and nature of this, we will let them be until further action is needed.

## Table Joining

Before we join the tables, notice that there are 13600 bills and 3400 clinical data. It seems that there is a pattern here. We will take a further look before joining any tables.


```python
print(len(bill_id))
print(len(clinical_data))
```

    13600
    3400
    

We check the number of bills for each patient on each visit.


```python
bill_id.groupby(['patient_id', 'date_of_admission']).count()['bill_id'].unique()
```




    array([4], dtype=int64)



It seems that each patient on each visit received 4 bills. As there is nothing to differentiate the bills other than the amount, we will combine the 4 bill amounts into 1 total bill for each patient on each visit. To do this, we combine `bill_id` and `bill_amount`, then we consolidate the amounts.


```python
df = bill_id.merge(bill_amount,  how='left',  on='bill_id').groupby(['patient_id', 'date_of_admission'])['amount'].sum().reset_index().rename(index=str, columns={"amount": "total_amount"})
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>date_of_admission</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00225710a878eff524a1d13be817e8e2</td>
      <td>2014-04-10</td>
      <td>5190.566695</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0029d90eb654699c18001c17efb0f129</td>
      <td>2012-11-07</td>
      <td>22601.497872</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0040333abd68527ecb53e1db9073f52e</td>
      <td>2013-01-19</td>
      <td>17447.181635</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00473b58e3dc8ae37b3cb34069705083</td>
      <td>2014-02-10</td>
      <td>15285.883220</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0078662d1d983dde68ea057c42d5b5cf</td>
      <td>2012-04-28</td>
      <td>73477.869010</td>
    </tr>
  </tbody>
</table>
</div>



Next, we combine the new table above with the other two tables. We join `df` with `clinical_data` on `patient_id` and `date_of_admission`, and we join it with `demographics` using just the `patient_id`.


```python
df = df.merge(clinical_data, 
              how='left', 
              left_on=['patient_id', 'date_of_admission'], 
              right_on=['id', 'date_of_admission']).drop('id',1).merge(demographics,
                                                                       how='left',
                                                                       on='patient_id')
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>date_of_admission</th>
      <th>total_amount</th>
      <th>date_of_discharge</th>
      <th>medical_history_1</th>
      <th>medical_history_2</th>
      <th>medical_history_3</th>
      <th>medical_history_4</th>
      <th>medical_history_5</th>
      <th>medical_history_6</th>
      <th>...</th>
      <th>symptom_5</th>
      <th>lab_result_1</th>
      <th>lab_result_2</th>
      <th>lab_result_3</th>
      <th>weight</th>
      <th>height</th>
      <th>gender</th>
      <th>race</th>
      <th>resident_status</th>
      <th>date_of_birth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00225710a878eff524a1d13be817e8e2</td>
      <td>2014-04-10</td>
      <td>5190.566695</td>
      <td>2014-04-22</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>13.4</td>
      <td>27.9</td>
      <td>96.0</td>
      <td>66.9</td>
      <td>155.0</td>
      <td>Female</td>
      <td>Chinese</td>
      <td>Singaporean</td>
      <td>1983-01-16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0029d90eb654699c18001c17efb0f129</td>
      <td>2012-11-07</td>
      <td>22601.497872</td>
      <td>2012-11-20</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>16.7</td>
      <td>26.5</td>
      <td>109.0</td>
      <td>89.1</td>
      <td>160.0</td>
      <td>Female</td>
      <td>Chinese</td>
      <td>Singaporean</td>
      <td>1943-10-14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0040333abd68527ecb53e1db9073f52e</td>
      <td>2013-01-19</td>
      <td>17447.181635</td>
      <td>2013-01-31</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>14.8</td>
      <td>25.2</td>
      <td>96.0</td>
      <td>79.5</td>
      <td>172.0</td>
      <td>Male</td>
      <td>Indian</td>
      <td>Singaporean</td>
      <td>1972-08-26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00473b58e3dc8ae37b3cb34069705083</td>
      <td>2014-02-10</td>
      <td>15285.883220</td>
      <td>2014-02-15</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>14.9</td>
      <td>28.7</td>
      <td>122.0</td>
      <td>81.1</td>
      <td>160.0</td>
      <td>Female</td>
      <td>Chinese</td>
      <td>Singaporean</td>
      <td>1976-07-23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0078662d1d983dde68ea057c42d5b5cf</td>
      <td>2012-04-28</td>
      <td>73477.869010</td>
      <td>2012-05-10</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>14.2</td>
      <td>27.2</td>
      <td>89.0</td>
      <td>74.7</td>
      <td>173.0</td>
      <td>Male</td>
      <td>Malay</td>
      <td>Foreigner</td>
      <td>1942-10-19</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



Since we have `date_of_discharge` and `date_of_admission`, we can compute some variables that may be useful. The first one is `lenght_of_stay`, which is the number of days the patient was hospitalised. The second one is `age`, which is the age of the patient during hospitalisation.


```python
df['length_of_stay'] = (df['date_of_discharge'] - df['date_of_admission']).dt.days
df['age'] = (df['date_of_admission'] - df['date_of_birth']).astype('timedelta64[Y]').astype('int64')
```

Afterwards, we set `patient_id` and `date_of_admission` as the indices for our table.


```python
df.set_index(['patient_id', 'date_of_admission'], inplace=True)
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>total_amount</th>
      <th>date_of_discharge</th>
      <th>medical_history_1</th>
      <th>medical_history_2</th>
      <th>medical_history_3</th>
      <th>medical_history_4</th>
      <th>medical_history_5</th>
      <th>medical_history_6</th>
      <th>medical_history_7</th>
      <th>preop_medication_1</th>
      <th>...</th>
      <th>lab_result_2</th>
      <th>lab_result_3</th>
      <th>weight</th>
      <th>height</th>
      <th>gender</th>
      <th>race</th>
      <th>resident_status</th>
      <th>date_of_birth</th>
      <th>length_of_stay</th>
      <th>age</th>
    </tr>
    <tr>
      <th>patient_id</th>
      <th>date_of_admission</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>00225710a878eff524a1d13be817e8e2</th>
      <th>2014-04-10</th>
      <td>5190.566695</td>
      <td>2014-04-22</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>27.9</td>
      <td>96.0</td>
      <td>66.9</td>
      <td>155.0</td>
      <td>Female</td>
      <td>Chinese</td>
      <td>Singaporean</td>
      <td>1983-01-16</td>
      <td>12</td>
      <td>31</td>
    </tr>
    <tr>
      <th>0029d90eb654699c18001c17efb0f129</th>
      <th>2012-11-07</th>
      <td>22601.497872</td>
      <td>2012-11-20</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>26.5</td>
      <td>109.0</td>
      <td>89.1</td>
      <td>160.0</td>
      <td>Female</td>
      <td>Chinese</td>
      <td>Singaporean</td>
      <td>1943-10-14</td>
      <td>13</td>
      <td>69</td>
    </tr>
    <tr>
      <th>0040333abd68527ecb53e1db9073f52e</th>
      <th>2013-01-19</th>
      <td>17447.181635</td>
      <td>2013-01-31</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>25.2</td>
      <td>96.0</td>
      <td>79.5</td>
      <td>172.0</td>
      <td>Male</td>
      <td>Indian</td>
      <td>Singaporean</td>
      <td>1972-08-26</td>
      <td>12</td>
      <td>40</td>
    </tr>
    <tr>
      <th>00473b58e3dc8ae37b3cb34069705083</th>
      <th>2014-02-10</th>
      <td>15285.883220</td>
      <td>2014-02-15</td>
      <td>1</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28.7</td>
      <td>122.0</td>
      <td>81.1</td>
      <td>160.0</td>
      <td>Female</td>
      <td>Chinese</td>
      <td>Singaporean</td>
      <td>1976-07-23</td>
      <td>5</td>
      <td>37</td>
    </tr>
    <tr>
      <th>0078662d1d983dde68ea057c42d5b5cf</th>
      <th>2012-04-28</th>
      <td>73477.869010</td>
      <td>2012-05-10</td>
      <td>0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>27.2</td>
      <td>89.0</td>
      <td>74.7</td>
      <td>173.0</td>
      <td>Male</td>
      <td>Malay</td>
      <td>Foreigner</td>
      <td>1942-10-19</td>
      <td>12</td>
      <td>69</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
#df.reset_index()[df.reset_index()['patient_id'].duplicated(keep=False)].set_index(['patient_id','date_of_admission']).iloc[:,3:10]
```

## Exploratory Data Analysis

In this section and the following sections, we will treat `total_amount` as the target variable with the other variables as its predictors. However, we will also explore any interesting pattern between the predictors. We start by taking a look at some properties of our data. 


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 3400 entries, (00225710a878eff524a1d13be817e8e2, 2014-04-10 00:00:00) to (ffd9644f8daf1d28493a7cd700bb30f4, 2013-06-04 00:00:00)
    Data columns (total 31 columns):
    total_amount          3400 non-null float64
    date_of_discharge     3400 non-null datetime64[ns]
    medical_history_1     3400 non-null int64
    medical_history_2     3167 non-null float64
    medical_history_3     3400 non-null int64
    medical_history_4     3400 non-null int64
    medical_history_5     3096 non-null float64
    medical_history_6     3400 non-null int64
    medical_history_7     3400 non-null int64
    preop_medication_1    3400 non-null int64
    preop_medication_2    3400 non-null int64
    preop_medication_3    3400 non-null int64
    preop_medication_4    3400 non-null int64
    preop_medication_5    3400 non-null int64
    preop_medication_6    3400 non-null int64
    symptom_1             3400 non-null int64
    symptom_2             3400 non-null int64
    symptom_3             3400 non-null int64
    symptom_4             3400 non-null int64
    symptom_5             3400 non-null int64
    lab_result_1          3400 non-null float64
    lab_result_2          3400 non-null float64
    lab_result_3          3400 non-null float64
    weight                3400 non-null float64
    height                3400 non-null float64
    gender                3400 non-null object
    race                  3400 non-null object
    resident_status       3400 non-null object
    date_of_birth         3400 non-null datetime64[ns]
    length_of_stay        3400 non-null int64
    age                   3400 non-null int64
    dtypes: datetime64[ns](2), float64(8), int64(18), object(3)
    memory usage: 871.7+ KB
    

Excluding the dates, itt seems that we have two types of predictors in our data: numerical variables and categorical variables.

### Numerical Variables

First of all, we take a look at all numerical variables. We compute the correlations between the predictors and show the heatmap of the computed correlation matrix.


```python
corr = df.drop('total_amount', axis=1).corr()
plt.figure(figsize=(12, 10))

sns.heatmap(corr[(corr >= 0.05) | (corr <= -0.05)], 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);
```


![png](output_64_0.png)


From the matrix above, we can see that the variables are uncorrelated of each other. With the exception of `weight` and `height` that have a correlation coefficient of `0.27`, all other variable pairs have correlation coefficients below `0.05` in absolute value.

Besides looking at the linear relationship between predictors, we also take a look at the linear relationship between `total_amount` and the predictors.


```python
top_corr = df.corr()['total_amount'][1:].abs().sort_values(ascending=False).index
top_corr = df.corr()['total_amount'][1:][top_corr]
top_corr
```




    symptom_5             0.516790
    age                   0.325586
    medical_history_1     0.226518
    symptom_3             0.183988
    symptom_2             0.157819
    weight                0.157511
    medical_history_6     0.141640
    symptom_4             0.130236
    symptom_1             0.128317
    medical_history_7     0.038951
    medical_history_5     0.036146
    medical_history_2     0.033965
    preop_medication_2    0.032263
    height                0.026057
    preop_medication_6    0.021606
    preop_medication_1    0.016112
    preop_medication_4    0.015196
    medical_history_3     0.011539
    medical_history_4    -0.009356
    length_of_stay        0.009006
    preop_medication_3    0.007839
    lab_result_1         -0.006518
    lab_result_2         -0.005537
    preop_medication_5    0.000453
    lab_result_3          0.000092
    Name: total_amount, dtype: float64



It looks like we have some variables that are correlated with `total_amount`. Of course, this result follows our intuition that an old and overweight patient with several symptoms or medical histories may need a more advanced procedure and care, thus increasing the bill amount.

Here we will plot `total_amount` against those correlated variables. To save space, we only plot the top 6 variables.


```python
# Q -> Q
features_to_analyse = [i for i in top_corr.index][:6]
features_to_analyse
```




    ['symptom_5', 'age', 'medical_history_1', 'symptom_3', 'symptom_2', 'weight']




```python
fig, ax = plt.subplots(2, 3, figsize = (18, 8))

for i, ax in enumerate(fig.axes):
    sns.regplot(x=features_to_analyse[i], y='total_amount', data=df, ax=ax)
```


![png](output_71_0.png)


From the plots above, we can see that `total_amount` increases as the variables increase. We suppose these variables will be important in predicting `total_amount`, but later on we will check them quantitatively using machine learning methods.

We can further separate the numerical variables in our data into two: continuous variables and binary variables. Here we will take a look at each of them.

#### Continuous Variables

First of all, we take a look at how the continuous variables are distributed with histograms.


```python
continuous_variables = ['age', 'height', 'weight', 'lab_result_1', 'lab_result_2', 'lab_result_3', 'length_of_stay']
```


```python
df[continuous_variables+ ['total_amount']].hist(figsize=(16,10), bins=20, xlabelsize=8, ylabelsize=8, alpha=0.8);
```


![png](output_77_0.png)


It seems that the most variables are centered and have Gaussian-shaped distributions. However, the distribution of `total_amount` seems to be right-skewed, and `age` with `height` seems to follow bimodal distributions.

#### Binary Variables

As for the binary variables, bar plots would be more appropriate.


```python
binary_variables = list(df)[2:20]
```


```python
fig, ax = plt.subplots(3,1,figsize = (18, 15))
sns.countplot(x="variable", hue="value", data=pd.melt(df[binary_variables[:7]]), ax=ax[0], alpha=0.7); # medical_history
sns.countplot(x="variable", hue="value", data=pd.melt(df[binary_variables[7:13]]), ax=ax[1], alpha=0.7); # preop_medication
sns.countplot(x="variable", hue="value", data=pd.melt(df[binary_variables[13:]]), ax=ax[2], alpha=0.7); # symptom
```


![png](output_82_0.png)


As we can see, although some variables are distributed equally, there are some variables that are not proportionate in size. This is expected, though. We wouldn't expect half of the patients to have a certain medical history, would we? However, just to make sure the variances are not near zero, we will compute them.


```python
df[binary_variables].agg([np.mean, np.var]).T
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>var</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>medical_history_1</th>
      <td>0.169118</td>
      <td>0.140558</td>
    </tr>
    <tr>
      <th>medical_history_2</th>
      <td>0.311336</td>
      <td>0.214473</td>
    </tr>
    <tr>
      <th>medical_history_3</th>
      <td>0.136176</td>
      <td>0.117667</td>
    </tr>
    <tr>
      <th>medical_history_4</th>
      <td>0.052059</td>
      <td>0.049363</td>
    </tr>
    <tr>
      <th>medical_history_5</th>
      <td>0.063953</td>
      <td>0.059883</td>
    </tr>
    <tr>
      <th>medical_history_6</th>
      <td>0.254706</td>
      <td>0.189887</td>
    </tr>
    <tr>
      <th>medical_history_7</th>
      <td>0.254412</td>
      <td>0.189742</td>
    </tr>
    <tr>
      <th>preop_medication_1</th>
      <td>0.503824</td>
      <td>0.250059</td>
    </tr>
    <tr>
      <th>preop_medication_2</th>
      <td>0.591176</td>
      <td>0.241758</td>
    </tr>
    <tr>
      <th>preop_medication_3</th>
      <td>0.820882</td>
      <td>0.147078</td>
    </tr>
    <tr>
      <th>preop_medication_4</th>
      <td>0.523235</td>
      <td>0.249534</td>
    </tr>
    <tr>
      <th>preop_medication_5</th>
      <td>0.819706</td>
      <td>0.147832</td>
    </tr>
    <tr>
      <th>preop_medication_6</th>
      <td>0.744118</td>
      <td>0.190463</td>
    </tr>
    <tr>
      <th>symptom_1</th>
      <td>0.619706</td>
      <td>0.235740</td>
    </tr>
    <tr>
      <th>symptom_2</th>
      <td>0.662353</td>
      <td>0.223707</td>
    </tr>
    <tr>
      <th>symptom_3</th>
      <td>0.544706</td>
      <td>0.248074</td>
    </tr>
    <tr>
      <th>symptom_4</th>
      <td>0.726471</td>
      <td>0.198770</td>
    </tr>
    <tr>
      <th>symptom_5</th>
      <td>0.526765</td>
      <td>0.249357</td>
    </tr>
  </tbody>
</table>
</div>



Fortunately, even the most unbalanced variable (`medical_history_4`) still has distribution ratio of 1:20. This doesn't necessarily mean problem, but we will just take a note of this.

### Categorical Variables

As for our categorical variables, we use box plots and bar plots to visualize the distribution of `total_amount` across the categories.


```python
categorical_variables = ['gender', 'race', 'resident_status']
```


```python
fig, ax = plt.subplots(2, 3, figsize = (18, 12))
boxplot_order = [['Female', 'Male'],
                 ['Chinese', 'Indian', 'Malay', 'Others'],
                 ['Singaporean', 'PR', 'Foreigner']]

for i, ax in enumerate(fig.axes):
    if i < 3:
        ax = sns.boxplot(x=categorical_variables[i], y='total_amount', data=df, ax=ax, order=boxplot_order[i])
        plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
    else:
        ax = sns.countplot(x=categorical_variables[i-3], alpha=0.7, data=df, ax=ax, order=boxplot_order[i-3])
```


![png](output_89_0.png)


From the `gender` variable, it seems that there is no visible discrepancies between the amount charged to females and males. It could be the case that this particular condition affects both genders equally. It could also be a sign that patients are not charged differently according to their gender.

As for `race`, it seems that Indian and Malay patients spent more than the others. It could be caused by the medical histories and symptoms that are more prominent in a certain race. Lastly, the `resident_status` of the patient. Obviously, this is a cost-driving factor that affects the cost through the pricing policy according the resident status. Singaporean are charged less than PRs and foreigners.

Lastly, the `resident_status` of the patient. Obviously, this is a cost-driving factor that affects the cost through the pricing policy according the resident status. Singaporean are charged less than PRs and foreigners.

An important thing to note is that most of the patients are Singaporean, with the majority of it are Chinese.

To end our exploratory data analysis, we show the average values of the binary variables across different categorical variables. Just skimming through the table, it seems there are only a few variables that differ across certain categories. For example, the percentage of patient with `medical_history_3` is higher in PR and Singaporean than that in Foreigner. Of course, numerical tests can be performed to verify this.


```python
pd.concat({'Gender': pd.DataFrame.pivot_table(df, values = binary_variables, columns = 'gender'),
           'Race': pd.DataFrame.pivot_table(df, values = binary_variables, columns = 'race'),
           'Resident Status': pd.DataFrame.pivot_table(df, values = binary_variables, columns = 'resident_status')},
          axis=1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">Gender</th>
      <th colspan="4" halign="left">Race</th>
      <th colspan="3" halign="left">Resident Status</th>
    </tr>
    <tr>
      <th></th>
      <th>Female</th>
      <th>Male</th>
      <th>Chinese</th>
      <th>Indian</th>
      <th>Malay</th>
      <th>Others</th>
      <th>Foreigner</th>
      <th>PR</th>
      <th>Singaporean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>medical_history_1</th>
      <td>0.175088</td>
      <td>0.163133</td>
      <td>0.167590</td>
      <td>0.156977</td>
      <td>0.172560</td>
      <td>0.196721</td>
      <td>0.180124</td>
      <td>0.159223</td>
      <td>0.170338</td>
    </tr>
    <tr>
      <th>medical_history_2</th>
      <td>0.316256</td>
      <td>0.306431</td>
      <td>0.304391</td>
      <td>0.336449</td>
      <td>0.316514</td>
      <td>0.327273</td>
      <td>0.315789</td>
      <td>0.290795</td>
      <td>0.314939</td>
    </tr>
    <tr>
      <th>medical_history_3</th>
      <td>0.133960</td>
      <td>0.138398</td>
      <td>0.131117</td>
      <td>0.139535</td>
      <td>0.155587</td>
      <td>0.114754</td>
      <td>0.074534</td>
      <td>0.139806</td>
      <td>0.139134</td>
    </tr>
    <tr>
      <th>medical_history_4</th>
      <td>0.056404</td>
      <td>0.047703</td>
      <td>0.053093</td>
      <td>0.066860</td>
      <td>0.048091</td>
      <td>0.027322</td>
      <td>0.037267</td>
      <td>0.046602</td>
      <td>0.053965</td>
    </tr>
    <tr>
      <th>medical_history_5</th>
      <td>0.071705</td>
      <td>0.056202</td>
      <td>0.066531</td>
      <td>0.053797</td>
      <td>0.066770</td>
      <td>0.041916</td>
      <td>0.075342</td>
      <td>0.078224</td>
      <td>0.060557</td>
    </tr>
    <tr>
      <th>medical_history_6</th>
      <td>0.235605</td>
      <td>0.273852</td>
      <td>0.256233</td>
      <td>0.258721</td>
      <td>0.253182</td>
      <td>0.234973</td>
      <td>0.229814</td>
      <td>0.279612</td>
      <td>0.251468</td>
    </tr>
    <tr>
      <th>medical_history_7</th>
      <td>0.271445</td>
      <td>0.237338</td>
      <td>0.258079</td>
      <td>0.267442</td>
      <td>0.234795</td>
      <td>0.262295</td>
      <td>0.298137</td>
      <td>0.285437</td>
      <td>0.245962</td>
    </tr>
    <tr>
      <th>preop_medication_1</th>
      <td>0.507638</td>
      <td>0.500000</td>
      <td>0.508772</td>
      <td>0.508721</td>
      <td>0.486563</td>
      <td>0.502732</td>
      <td>0.447205</td>
      <td>0.493204</td>
      <td>0.509178</td>
    </tr>
    <tr>
      <th>preop_medication_2</th>
      <td>0.595770</td>
      <td>0.586572</td>
      <td>0.578486</td>
      <td>0.630814</td>
      <td>0.605375</td>
      <td>0.612022</td>
      <td>0.534161</td>
      <td>0.578641</td>
      <td>0.596916</td>
    </tr>
    <tr>
      <th>preop_medication_3</th>
      <td>0.815511</td>
      <td>0.826266</td>
      <td>0.822715</td>
      <td>0.840116</td>
      <td>0.817539</td>
      <td>0.775956</td>
      <td>0.869565</td>
      <td>0.800000</td>
      <td>0.821953</td>
    </tr>
    <tr>
      <th>preop_medication_4</th>
      <td>0.541716</td>
      <td>0.504711</td>
      <td>0.516159</td>
      <td>0.529070</td>
      <td>0.548798</td>
      <td>0.497268</td>
      <td>0.534161</td>
      <td>0.549515</td>
      <td>0.517621</td>
    </tr>
    <tr>
      <th>preop_medication_5</th>
      <td>0.830787</td>
      <td>0.808598</td>
      <td>0.823638</td>
      <td>0.811047</td>
      <td>0.814710</td>
      <td>0.808743</td>
      <td>0.795031</td>
      <td>0.834951</td>
      <td>0.818282</td>
    </tr>
    <tr>
      <th>preop_medication_6</th>
      <td>0.739718</td>
      <td>0.748528</td>
      <td>0.741921</td>
      <td>0.741279</td>
      <td>0.752475</td>
      <td>0.743169</td>
      <td>0.788820</td>
      <td>0.724272</td>
      <td>0.745228</td>
    </tr>
    <tr>
      <th>symptom_1</th>
      <td>0.615746</td>
      <td>0.623675</td>
      <td>0.620499</td>
      <td>0.598837</td>
      <td>0.616690</td>
      <td>0.661202</td>
      <td>0.577640</td>
      <td>0.609709</td>
      <td>0.624082</td>
    </tr>
    <tr>
      <th>symptom_2</th>
      <td>0.670388</td>
      <td>0.654299</td>
      <td>0.656510</td>
      <td>0.697674</td>
      <td>0.660537</td>
      <td>0.672131</td>
      <td>0.695652</td>
      <td>0.664078</td>
      <td>0.660059</td>
    </tr>
    <tr>
      <th>symptom_3</th>
      <td>0.528790</td>
      <td>0.560660</td>
      <td>0.544321</td>
      <td>0.584302</td>
      <td>0.541726</td>
      <td>0.486339</td>
      <td>0.490683</td>
      <td>0.557282</td>
      <td>0.545521</td>
    </tr>
    <tr>
      <th>symptom_4</th>
      <td>0.716804</td>
      <td>0.736160</td>
      <td>0.725762</td>
      <td>0.706395</td>
      <td>0.735502</td>
      <td>0.737705</td>
      <td>0.708075</td>
      <td>0.718447</td>
      <td>0.729075</td>
    </tr>
    <tr>
      <th>symptom_5</th>
      <td>0.519976</td>
      <td>0.533569</td>
      <td>0.520776</td>
      <td>0.587209</td>
      <td>0.519095</td>
      <td>0.513661</td>
      <td>0.546584</td>
      <td>0.526214</td>
      <td>0.525698</td>
    </tr>
  </tbody>
</table>
</div>



Just looking at it, we can see that some binary variables are more pronounced in a certain gender, race, or resident status (this could be caused by difference in lifestyle, assuming the foreigners and PRs reside in Singapore).

## Machine Learning

In this section, we will try some machine learning methods to predict `total_amount`. We will use the mean squared error (MSE) to measure the performance of our models.

First of all, we create the design matrix. In order to do so, we transform the categorical variables into dummy variables.


```python
X = pd.concat([df[continuous_variables+binary_variables],
           pd.get_dummies(df['gender']).drop('Female', 1), 
           pd.get_dummies(df['race']).drop('Others', 1), 
           pd.get_dummies(df['resident_status']).drop('Foreigner', 1) ], axis=1)

y = df['total_amount']
```

Then, we scale our data to have mean `0` and variance `1`. As for the binary variables, we don't standardize them to ease the interpretation.


```python
X[continuous_variables] = (X[continuous_variables] - X[continuous_variables].mean()) / X[continuous_variables].std()
```

To deal with missing data in `medical_history_2` and `medical_history_5`, since we don't know the nature of those missing values, we will fill them with `0`. We expect that the missing values mean so. Besides, the majority of those variables are `0`.


```python
X.fillna(0, inplace=True)
```

Lastly, we split our data into training data and test data. We will do a 2:1 split and store them into different objects.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

### Linear Regression

We start with a simple linear regression, or the ordinary least square (OLS). Since this model doesn't need any parameter tuning, we will make this our baseline method.


```python
from sklearn import linear_model
ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



We then take a look at the fitted coefficients.


```python
pd.Series(ols.coef_, index = list(X)).sort_values(ascending=False)
```




    symptom_5             10492.536881
    Malay                  8208.713402
    medical_history_1      6166.795082
    symptom_3              3983.479633
    symptom_2              3670.804277
    medical_history_6      3497.629477
    age                    3129.440722
    symptom_4              3087.384790
    symptom_1              2756.801696
    medical_history_5      1624.066874
    weight                 1590.097895
    Indian                 1589.934194
    medical_history_7      1028.745463
    preop_medication_1      574.077724
    medical_history_4       434.190073
    medical_history_2       430.687939
    preop_medication_2      360.444001
    preop_medication_5      327.449013
    medical_history_3       254.078566
    preop_medication_3      251.787988
    preop_medication_4       27.726127
    lab_result_2             21.208274
    preop_medication_6       12.013876
    length_of_stay           -1.537716
    Male                     -8.331251
    lab_result_3            -48.636477
    lab_result_1            -52.178155
    height                 -278.949102
    Chinese               -2364.513823
    PR                   -17259.523883
    Singaporean          -21253.383500
    dtype: float64



From the sorted coefficients above, we can infer that if a patient shows `symptom_5`, the patient is expected to be charged around 10000 SGD more. Similarly, if a patient has `medical_history_1`, the patient is expected to be charged around 6000 SGD more. Generally, the symptom variables drive the cost the most, with some medical history variables also affecting the cost in the thousands. On the contrary, some of the medical history variables and all pre-op medications and lab results variables only affect the cost in the hundreds.

Besides the clinical data variables, we can see that some physical variables such as `age` and `weight` have their place in driving `total_amount`. 

An interesting observation comes from the dummy variables from `resident_status`. From the coefficients, we can see that foreigners pay a premium of more than 20000 SGD, twice the impact of `symptom_5`. 

Besides `resident_status`, `race` also plays a significant part in driving the cost. Malays are expected to have higher costs, while Chinese are expected to have lower costs. Perhaps further analysis can be done to see whether there's a difference between the medical conditions of Malay patients and those of Chinese patients. To give a quick illustration, here are the coefficients fitted only from Chinese Singaporean patients.


```python
ols.intercept_
```




    23587.448824204963




```python
chi_sg = (X_train['Singaporean']==1) & (X_train['Chinese']==1)
pd.Series(linear_model.LinearRegression().fit(X_train[chi_sg],
                                              y_train[chi_sg]).coef_, index = list(X)).sort_values(ascending=False)
```




    symptom_5             8356.696313
    medical_history_1     4826.488952
    symptom_3             3197.598497
    symptom_2             2944.616646
    medical_history_6     2845.792355
    age                   2572.255885
    symptom_4             2474.955079
    symptom_1             2082.526971
    weight                1271.063769
    medical_history_5      848.670941
    medical_history_7      761.211624
    preop_medication_1     467.793345
    medical_history_3      288.406603
    preop_medication_3     284.495389
    preop_medication_5     258.610569
    medical_history_2      255.205626
    preop_medication_6     199.942166
    preop_medication_2     178.460944
    medical_history_4      168.989977
    Male                    29.785667
    lab_result_1            28.099125
    PR                       0.000000
    Chinese                  0.000000
    Indian                   0.000000
    Malay                    0.000000
    Singaporean              0.000000
    length_of_stay          -2.014334
    lab_result_3           -12.773178
    lab_result_2           -13.761169
    preop_medication_4     -67.042904
    height                -236.964258
    dtype: float64



After removing patients that are not Chinese Singaporean, we still have a similar result from the previous one. Of course, this model should work better on Chinese Singaporean patient, since we have removed irrelevant noises from our data. The same approach, however, may not be applicable on other race or resident status. For example, there are only 43 Indian PRs on our data. For this reason, we will stick to the first model.


```python
sum((X['PR']==1) & (X['Indian']==1))
```




    43



Now it's time to test our model. Here we predict `total_amount` of the test data with our model. Then, we compute the MSE.


```python
((ols.predict(X_test) - y_test) ** 2).mean() ** 0.5
```




    2473.7930898002946



Not too bad for a baseline model. To give a better illustration on the residual distribution, we will plot the residuals against the predicted values.


```python
sns.residplot(x=ols.predict(X_test),
              y=ols.predict(X_test) - y_test,
              lowess=True,
              scatter_kws={'alpha': 0.5}, 
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
```


![png](output_120_0.png)


From the plot above, we see that the residuals are close to `0` around 25000, with a decreasing tendency as it deviates from that point. The residual plot also resembles a quadratic function. Perhaps we can try polynomial regression?

### Polynomial Regression


```python
from sklearn.preprocessing import PolynomialFeatures
X_train_quad = PolynomialFeatures(degree=2).fit_transform(X_train)
X_test_quad = PolynomialFeatures(degree=2).fit_transform(X_test)

poly = linear_model.LinearRegression(fit_intercept=False).fit(X_train_quad, y_train)
sns.residplot(x=poly.predict(X_test_quad),
              y=poly.predict(X_test_quad) - y_test,
              lowess=True,
              scatter_kws={'alpha': 0.5}, 
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
```


![png](output_123_0.png)



```python
((poly.predict(X_test_quad) - y_test) ** 2).mean() ** 0.5
```




    520.6333942565653



It seems that adding polynomial features increases the performance dramatically. We have reduced the MSE to a mere 500, a very acceptable error rate, considering the magnitude of `total_amount`. However, by doing so we have increased the number of variables into more than 500!

Here we will see the coefficients.


```python
poly_var = PolynomialFeatures(degree=2).fit(X_train).get_feature_names(X_train.columns)

# the where method is to filter out coefficients with 
# very large magnitude that are not meaningful (they cancel out each other automatically)
pd.Series(poly.coef_, index=poly_var).where(lambda x : abs(x) <= 1e6).dropna().sort_values(ascending=False)
```




    1                                  5639.613742
    symptom_5 Malay                    3939.218262
    age                                3339.199080
    medical_history_1 Malay            2118.057861
    weight                             1712.644881
    Chinese Singaporean                1674.812866
    symptom_3 Malay                    1469.509766
    age symptom_5                      1462.501465
    symptom_2 Malay                    1445.858887
    medical_history_6 Malay            1351.910400
    Chinese PR                         1323.869507
    symptom_4 Malay                    1164.509766
    age Malay                          1134.828209
    symptom_1 Malay                    1125.597168
    medical_history_4 PR               1116.309006
    medical_history_4 Singaporean      1028.158813
    preop_medication_6 PR               951.509521
    preop_medication_6 Singaporean      882.907227
    age medical_history_1               867.558167
    symptom_5 Indian                    781.081055
    weight symptom_5                    751.161133
    weight Malay                        691.657959
    medical_history_7 Malay             655.600586
    age medical_history_6               511.041260
    age symptom_2                       502.769043
    age symptom_3                       482.698730
    age symptom_4                       470.556030
    medical_history_1 Indian            468.994568
    medical_history_5 Malay             458.911194
    age symptom_1                       384.911377
                                          ...     
    medical_history_1 Chinese          -707.257141
    preop_medication_1 PR              -748.192017
    preop_medication_1 Singaporean     -880.754395
    medical_history_7 PR              -1115.694702
    symptom_5 Chinese                 -1121.798340
    weight PR                         -1289.413971
    medical_history_7 Singaporean     -1293.534180
    medical_history_5 PR              -1370.958923
    weight Singaporean                -1555.666290
    medical_history_5 Singaporean     -1651.486816
    medical_history_6 PR              -2139.431152
    symptom_4 PR                      -2241.316895
    age PR                            -2350.048340
    symptom_2 PR                      -2734.636719
    medical_history_6 Singaporean     -2757.796249
    symptom_4 Singaporean             -2796.583984
    symptom_1 PR                      -2836.521729
    age Singaporean                   -2897.429077
    Indian PR                         -3084.605865
    symptom_2 Singaporean             -3342.952393
    symptom_1 Singaporean             -3384.825684
    symptom_3 PR                      -3399.031128
    Indian Singaporean                -3552.968750
    symptom_3 Singaporean             -4076.969482
    medical_history_1 PR              -4649.243469
    medical_history_1 Singaporean     -5714.288147
    Malay PR                          -7799.527100
    symptom_5 PR                      -8168.816528
    Malay Singaporean                 -9290.119629
    symptom_5 Singaporean            -10122.353271
    Length: 480, dtype: float64



As expected, interactions between the influential variables from the OLS model dominate both end of the coefficients range. Perhaps we can conclude that this model is very strong in predictive power, but weak in simplicity and interpretability. It's far more easier to explain the first model to general public or stakeholders.

### LASSO

To extend our approach with regression model, we also try regularization method with LASSO. Because LASSO can set a coefficient as `0`, this approach can also double as feature selection.

Here we use cross-validation to fine tune the parameter `alpha`.


```python
from sklearn.model_selection import cross_val_score, GridSearchCV
```


```python
from sklearn import linear_model
lassoCV = linear_model.LassoCV()
```


```python
alpha = lassoCV.fit(X_train, y_train).alpha_
print('Best alpha:',alpha) 
```

    Best alpha: 3.62119261889
    

It seems that the best `alpha` is small. We may expect the model to still keep many features.


```python
lasso = linear_model.Lasso(alpha=alpha).fit(X_train, y_train)
pd.Series(lasso.coef_, index=list(X)).sort_values(ascending=False)
```




    symptom_5             10479.944277
    Malay                  8108.140868
    medical_history_1      6137.249266
    symptom_3              3967.954046
    symptom_2              3654.051362
    medical_history_6      3471.640016
    age                    3127.416869
    symptom_4              3067.034204
    symptom_1              2741.664311
    weight                 1586.034943
    medical_history_5      1564.930888
    Indian                 1470.527868
    medical_history_7      1010.169387
    preop_medication_1      554.504465
    medical_history_2       415.760583
    medical_history_4       356.835859
    preop_medication_2      341.520193
    preop_medication_5      306.479294
    preop_medication_3      230.891445
    medical_history_3       218.631492
    lab_result_2             16.092360
    preop_medication_4       16.005380
    preop_medication_6        0.000000
    Male                     -0.000000
    length_of_stay           -0.000000
    lab_result_3            -44.900955
    lab_result_1            -47.852770
    height                 -277.414161
    Chinese               -2441.951953
    PR                   -17076.859503
    Singaporean          -21091.064102
    dtype: float64



As expected, the model only remove 3 variables. Let's take a look at the test error and the residual plot.


```python
((lasso.predict(X_test) - y_test) ** 2).mean() ** 0.5
```




    2466.8972015320073




```python
sns.residplot(x=lasso.predict(X_test),
              y=lasso.predict(X_test) - y_test,
              lowess=True,
              scatter_kws={'alpha': 0.5}, 
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
```


![png](output_137_0.png)


Unfortunately, it's only a slight improvement from the OLS. It's expected since the optimal `alpha` itself is small. The residual plot itself looks very similar to that of the OLS. From this result, we can say that the more sparse model from larger `alpha` will not produce better result, thus it's better to keep most of the variables and apply only a bit of regularization penalty.

### Regression Tree

Besides least square methods, tree methods can also be used. One strong reason to use tree method is the interpretability. Here we will try the regression tree method on our data and see its performance.

For this method, we will use cross-validation to find the optimal depth limit of the tree.


```python
from sklearn import tree

param_candidates = [
  {'max_depth': [1, 2, 4, 6, 8, 10]},
]
regtCV = GridSearchCV(estimator=tree.DecisionTreeRegressor(), param_grid = param_candidates).fit(X_train, y_train)
```


```python
print('Best max_depth:',regtCV.best_estimator_.max_depth) 
```

    Best max_depth: 8
    

It seems that the data favour a deep tree. In terms of interpretability, this kind of tree may fare badly. Let's see the numerical performance, though.


```python
regt = regtCV.best_estimator_.fit(X_train, y_train)
```


```python
((regt.predict(X_test) - y_test) ** 2).mean() ** 0.5
```




    4868.978541390168




```python
sns.residplot(x=regt.predict(X_test),
              y=regt.predict(X_test) - y_test,
              lowess=True,
              scatter_kws={'alpha': 0.5}, 
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
```


![png](output_146_0.png)


It seems that the tree model is not suitable for our data. We will just try other methods.

### Support Vector Regression

Here we will try the support vector method to fit our data.

Similar to the tree method above, we will also use cross-validation to tune the parameters and choose the kernel.


```python
from sklearn import svm

param_candidates = [
  {'C': [1, 10, 100, 10000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 10000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']},
]
svrCV = GridSearchCV(estimator=svm.SVR(), param_grid = param_candidates).fit(X_train, y_train)
```


```python
print('Best C:',svrCV.best_estimator_.C) 
print('Best Kernel:',svrCV.best_estimator_.kernel)
print('Best Gamma:',svrCV.best_estimator_.gamma)
```

    Best C: 10000
    Best Kernel: linear
    Best Gamma: auto
    

It seems that linear kernel is better for our application. We will see the numerical performance.


```python
svr = svrCV.best_estimator_.fit(X_train, y_train)
```


```python
((svr.predict(X_test) - y_test) ** 2).mean() ** 0.5
```




    2555.0644324467503




```python
sns.residplot(x=svrCV.predict(X_test),
              y=svrCV.predict(X_test) - y_test,
              lowess=True,
              scatter_kws={'alpha': 0.5}, 
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
```


![png](output_155_0.png)


Apparently, support vector regression is not better than OLS in our application.

### Neural Network

The last method we will try is the neural network. Neural network is known as a very powerful black box.

Here we will just try a simple neural network with up to two hidden layers. To select the number of nodes in the hidden layers, we use cross-validation to try several values.


```python
from sklearn.neural_network import MLPRegressor
```


```python
param_candidates = [
  {'hidden_layer_sizes': [(2), (5), (7), (10), (10,2), (10,5), (10,10)]},
]
nnetCV = GridSearchCV(estimator=MLPRegressor(solver='lbfgs', activation='relu'), param_grid = param_candidates).fit(X_train, y_train)
```


```python
nnetCV.best_params_
```




    {'hidden_layer_sizes': (10, 2)}



It seems that we will use a neural network model with 2 hidden layers of 10 nodes each.


```python
nnetCV.best_estimator_.fit(X_train, y_train)
```




    MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(10, 2), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=None,
           shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1,
           verbose=False, warm_start=False)




```python
((nnetCV.predict(X_test) - y_test) ** 2).mean() ** 0.5
```




    1122.3749089181533




```python
sns.residplot(x=nnetCV.predict(X_test),
              y=nnetCV.predict(X_test) - y_test,
              lowess=True,
              scatter_kws={'alpha': 0.5}, 
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
```


![png](output_165_0.png)


We can see that this model performs quite well. Although the polynomial regression is better, this model provides significant improvement over the OLS. Perhaps more exploration can be done to further fine tune the model.

## Conclusion

In this analysis, we analysed clinical and financial data of patients with a certain condition. The goal is to predict the cost charged to a patient using the patient's clinical data. Before the analysis, we cleaned the data to make our analysis easier. Then, we did some exploratory data analysis to get a better look at our data. Furthermore, we tried several machine learning methods to predict the cost.

From our analysis, we see that the symptom variables and the demographic variables (for example, race and resident status) affect the cost more. On the other hand, a majority of the medical history variables, all pre-op medications, and all lab results variables don't seem to contribute much in determining the cost. These patterns can be seen in both our exploratory data analysis part and the statistical modeling part.

We also found that simple model with heavily regularised parameters are not suitable for our data. In our analysis, complicated methods with many parameters such as polynomial regression and neural network perform the best, while simple methods don't.

To close our analysis, we hope that this analysis may be of value to parties from various background, be it someone who wants to estimate his/her expected hospitalisation cost, an insurance company who wants to compute premiums for their prospective clients, or a digital health company, focused on solving complex problems in healthcare.
