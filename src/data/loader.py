import pandas as pd

class DataLoader:
    
    def init(self, data_dir):
        self.data_dir = data_dir
        
    def load_generation(self):
        return pd.read_csv(
            self.data_dir /
            "Solar_Energy_Generation.csv"
        )
    
    def load_weather(self):
        return pd.read_csv(
            self.data_dir /
            "Weather_Data_reordered_all.csv"
        )
    
    def load_irridance(self):
        return pd.read_csv(
            self.data_dir /
            "Solar_Irradiance.csv"
        )
    
    def load_sites(self):
        return pd.read_csv(
            self.data_dir /
            "Solar_Site_Details.csv"
        )