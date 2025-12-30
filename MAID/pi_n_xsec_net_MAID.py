import numpy as np
import pandas as pd
from basic.pi_n_xsec_net import TrainConfig, PiPlusNElectroproductionRegressor

class PiPlusNElectroproductionRegressorMAID(PiPlusNElectroproductionRegressor):
    LABEL_COLUMN = 'dsigma_dOmega_maid'
    DATA_COLUMNS = ["id", "Ebeam", "W", "Q2", "cos_theta", "phi", "dsigma_dOmega",
                    "error", "weight", "dsigma_dOmega_maid"]
    def load_and_prepare_dataframe(self) -> pd.DataFrame:
        if '.txt' in self.cfg.data_path:
            df = pd.read_csv(self.cfg.data_path, delimiter="\t", header=None)
        else:
            df = pd.read_csv(self.cfg.data_path, delimiter=",")
        df.columns = self.DATA_COLUMNS

        df.loc[self.cfg.ebeam_fix_from:self.cfg.ebeam_fix_to, "Ebeam"] = self.cfg.ebeam_fix_value

        df["cos_phi"] = np.cos(df["phi"])

        df = df.iloc[df[self.FEATURE_COLUMNS].drop_duplicates().index]
        df = df.drop(columns=["id"])

        q = self.cfg.clip_quantile
        df = df[df[self.LABEL_COLUMN] <= df[self.LABEL_COLUMN].quantile(q)]
        df = df[df["error"] <= df["error"].quantile(q)]

        return df.reset_index(drop=True)


def main():
    cfg = TrainConfig()
    cfg.data_path = '/Users/golda/Documents/Study/pi_n_xsec/data/MAID/df_maid_as_exp.csv'
    cfg.run_name+='_MAID'

    model = PiPlusNElectroproductionRegressorMAID(cfg)
    metrics = model.fit()
    print(f"Done. Test MAE = {metrics['mae']:.6f}, Test MSE = {metrics['mse']:.6f}")

if __name__ == "__main__":
    main()
