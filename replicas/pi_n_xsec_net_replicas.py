import tqdm
import numpy as np
import pandas as pd
from basic.pi_n_xsec_net import TrainConfig, PiPlusNElectroproductionRegressor

def generate_grid(df, step_W=0.005, step_Q2=0.1, step_cos_theta=0.1, step_phi=0.05):
    E_range = df.Ebeam.unique().tolist()

    data_grid = []
    for E in tqdm.tqdm(E_range):
        W_min = df[df.Ebeam==E].W.min() - 0.1
        W_max = df[df.Ebeam==E].W.max() + 0.1 + step_W

        Q2_min = df[df.Ebeam==E].Q2.min() - 0.1
        Q2_max = df[df.Ebeam==E].Q2.max() + 0.1 + step_Q2

        for W in np.arange(W_min, W_max, step_W):
            for Q2 in np.arange(Q2_min, Q2_max, step_Q2):
                 for cos_theta in np.arange(-1, 1, step_cos_theta):
                        for phi in np.arange(0, 2*np.pi, step_phi):
                            data_grid.append([E,W,Q2,cos_theta,phi])

    df_grid = pd.DataFrame(data_grid)
    df_grid.columns = ['Ebeam', 'W', 'Q2', 'cos_theta', 'phi']

    df_grid.W = np.round(df_grid.W, 3)
    df_grid.Q2 = np.round(df_grid.Q2, 3)
    df_grid.cos_theta = np.round(df_grid.cos_theta, 3)
    df_grid.phi = np.round(df_grid.phi, 3)
    df_grid['cos_phi'] = np.cos(df_grid.phi)
    return df_grid


class PiPlusNElectroproductionRegressorReplicas(PiPlusNElectroproductionRegressor):
    def load_and_prepare_dataframe(self):
        df = pd.read_csv(self.cfg.data_path, delimiter=",")
        columns = self.FEATURE_COLUMNS + [self.LABEL_COLUMN]
        df = df[columns]
        return df

def main():
    cfg = TrainConfig()

    cfg.es_patience = 5
    cfg.wandb_project += '_REPLICAS_UNIFIED'
    cfg.run_name+='_REPLICAS_UNIFIED'

    grid_final_path = '/Users/golda/Documents/Study/pi_n_xsec/data/replicas/df_replicas_grid_final.csv'
    grid_intermediate_path = '/Users/golda/Documents/Study/pi_n_xsec/data/replicas/df_replicas_grid.csv'
    final_path = '/Users/golda/Documents/Study/pi_n_xsec/data/replicas/df_replicas_final.csv'
    intermediate_path = '/Users/golda/Documents/Study/pi_n_xsec/data/replicas/df_replicas.csv'

    # df = pd.read_csv(intermediate_path)

    model = PiPlusNElectroproductionRegressor(cfg)
    metrics = model.fit()

    print(f"Done. Test MAE = {metrics['mae']:.6f}, Test MSE = {metrics['mse']:.6f}")

    df = model.load_and_prepare_dataframe()
    df['dsigma_dOmega_predicted'] = model.predict_df(df)
    for it in range(100):
        df[f'dsigma_dOmega_replica_{it}'] = df.apply(lambda x: np.random.normal(loc=x.dsigma_dOmega_predicted, scale=x.error), axis=1)

    df.to_csv(intermediate_path)

    df_grid = generate_grid(df)

    df_grid.to_csv(grid_intermediate_path)

    for it in tqdm.tqdm(range(100)):
        cfg = TrainConfig()
        cfg.es_patience = 5
        cfg.wandb_project += '_REPLICAS_UNIFIED'
        cfg.run_name += f'_REPLICAS_{it}'

        cfg.data_path = intermediate_path
        cfg.phi_to_rad = False

        model = PiPlusNElectroproductionRegressorReplicas(cfg)

        model.LABEL_COLUMN = f"dsigma_dOmega_replica_{it}"
        model.DATA_COLUMNS = df.columns

        model.fit()
        df[f'dsigma_dOmega_replica_{it}_predicted'] = model.predict_df(df)
        df_grid[f'dsigma_dOmega_replica_{it}_predicted'] = model.predict_df(df_grid)

        if it % 5 == 0:
            df.to_csv(final_path)
            df_grid.to_csv(grid_final_path)
        else:
            pass

    df.to_csv(final_path)
    df_grid.to_csv(grid_final_path)


if __name__ == "__main__":
    main()
