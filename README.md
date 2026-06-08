# solar-performance-intelligence

## Dataset 
- Now we are making use of [UNISOLAR Dataset](https://www.kaggle.com/datasets/pythonafroz/solar-power-generation-and-energy-consumption-data/data) which contains high-granuality PV solar energy generation, solar irradiance, and weather data from 42 PV sites deployed across five campuses at La Trobe University, Victoria, and Australia.
- The dataset spans over 2 years from 2020-01-01 to 2022-04-24 with 15 minutes of interval between reading.

```text
.
├── archive
│   ├── building_consumption.csv
│   ├── building_meta.csv
│   ├── building_submeter_consumption.csv
│   ├── calender.csv
│   ├── campus_meta.csv
│   ├── events.csv
│   ├── gas_consumption.csv
│   ├── nmi_consumption.csv
│   ├── nmi_meta.csv
│   ├── water_consumption.csv
│   └── weather_data.csv
└── unisolar
    ├── Monthly_Summary_Solar.csv
    ├── Solar_Energy_Generation.csv
    ├── Solar_Irradiance.csv
    ├── Solar_Site_Details.csv
    └── Weather_Data_reordered_all.csv
```

- <b>columns of each CSVs</b>
    - archive/building_consumption.csv
        - campus_id
        - meter_id
        - timestamp
        - consumption

## Table Schema finally
| Feature             | Source       |
| ------------------- | ------------ |
| Timestamp           | Generation   |
| CampusKey           | Generation   |
| SiteKey             | Generation   |
| SolarGeneration     | Generation   |
| GHI                 | Irradiance   |
| CloudOpacity        | Irradiance   |
| AirTemperature      | Weather      |
| RelativeHumidity    | Weather      |
| WindSpeed           | Weather      |
| DewPointTemperature | Weather      |
| Latitude            | Site Details |
| Longitude           | Site Details |
| kWp                 | Site Details |
These are the features that are seem to be affecting the model as per the domain knowledge provided that weather parameters, geolocation, and solar parameters affect the generation effectiveness of solar in a specific location.
<br>

# Refernces
[1] S. Wimalaratne, D. Haputhanthri, S. Kahawala, G. Gamage, D. Alahakoon and A. Jennings, "UNISOLAR: An Open Dataset of Photovoltaic Solar Energy Generation in a Large Multi-Campus University Setting," 2022 15th International Conference on Human System Interaction (HSI), 2022, pp. 1-5, doi: 10.1109/HSI55341.2022.9869474.