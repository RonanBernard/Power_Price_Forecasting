

# XG Boost with lag hours of 3 days (25 - 96)

rmse = 14
mae = 7.3


13/08/2025
- V1 dataset created with only FR data

DatetimeIndex: 68864 entries, 2015-01-09 00:00:00+01:00 to 2024-12-31 23:00:00+01:00
Data columns (total 84 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   FR_Solar          68864 non-null  float64
 1   FR_Wind Offshore  68864 non-null  float64
 2   FR_Wind Onshore   68864 non-null  float64
 3   FR_forecast_load  68864 non-null  float64
 4   FR_price          68864 non-null  float64
 5   Day_sin           68864 non-null  float64
 6   Day_cos           68864 non-null  float64
 7   Year_sin          68864 non-null  float64
 8   Year_cos          68864 non-null  float64
 9   EUA_EUR           68864 non-null  float64
 10  TTF_EUR           68864 non-null  float64
 11  ARA_EUR           68864 non-null  float64
 12  price_lag_25h     68864 non-null  float64
 ...
  83  price_lag_96h     68864 non-null  float64

  
  16/08/2025
  The MPL model does not manage to generalize, the gap between training and validation loss remains very high, whatever regularization, dropout or simple model I use.
  This could be due to a lack of data in the past to forecast the future.

![Monthly French prices](docs/monthly_FR_price.png)

  Indeed the val set was composed of prices from 2021 to 2023 which were much higher than usual. So I removed month with 3-months-rolling average above 100 €/MWh. And now the model generalizes much better.

Final Training Metrics:
RMSE: 11.5553
['mae']: [6.47]

Final Validation Metrics:
RMSE: 19.0647
['mae']: [13.6979]


  Fuel prices were not included, they were removed in the method create_features. Results largely improved when they are included, see below.

Final Training Metrics:
RMSE: 10.5284
['mae']: [5.301108360290527]

Final Validation Metrics:
RMSE: 15.6954
['mae']: [10.463689804077148]



  17/08/2025
  Added crossborder flows from ENTSO-E

  The next dataset should contrain crossborder flows.

   18/08/2025
  Finished first draft of ATT model.   

First model without regularization
  Final Training Metrics:
RMSE: 8.7908
['mae']: [4.257368564605713]

Final Validation Metrics:
RMSE: 15.4332
['mae']: [9.882774353027344]


dropout=0.2, 
lambda_reg=1e-4

Final Training Metrics:
RMSE: 9.6352
['mae']: [4.796072006225586]

Final Validation Metrics:
RMSE: 17.4804
['mae']: [10.821349143981934]


20/08/2025
Correction on windows to get full days
Correction on the model to use ContextProcessor instead of custom layer

Final Training Metrics:
RMSE: 8.9187
['mae']: [3.711181163787842]

Final Validation Metrics:
RMSE: 40.1233
['mae']: [31.670751571655273]


21/08/2025
Created a simple LSTM model and got these results. The model has difficulty to generalize, and it does not seem to be linked to the model complexity. The issue might be coming from the lack of data to generalize.

Final Training Metrics:
RMSE: 6.8434
['mae', 'mape']: [3.692394733428955, 25403.365234375]

Final Validation Metrics:
RMSE: 30.1847
['mae', 'mape']: [21.525312423706055, 56601372.0]


Mixing the dataset, breaking chronology and allowing data leakage : the model learns well and is generalizing. 

![Results with data shuffle](docs/20250821_Results_Data_Shuffle.png)

So the issue comes indeed from the lack of information to generalize for future periods.


25/08/2025
Try rolling-horizon validation split to have the latest data in training to forecast val


