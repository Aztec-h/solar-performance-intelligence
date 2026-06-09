import pandas as pd

class DataLoader:
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def _load_csv(self, data_dir):
        filepath = self.data_dir / data_dir
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"File not found: {filepath}"
            )
        
        return pd.read_csv(filepath)
    
    def load_generation(self):
        return self._load_csv(
            "Solar_Energy_Generation.csv"
        )
        
    def load_weather(self):
        return self._load_csv(
            "Weather_Data_reordered_all.csv"
        )
    
    def load_irradiance(self):
        return self._load_csv(
            "Solar_Irradiance.csv"
        )
    
    def load_site_details(self):
        return self._load_csv(
            "Solar_Site_Details.csv"
        )
        
    def load_all(self):
        return {
            "generation": self.load_generation(),
            "weather": self.load_weather(),
            "irradiance": self.load_irradiance(),
            "sites": self.load_site_details()
        }