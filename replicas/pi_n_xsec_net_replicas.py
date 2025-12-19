from basic.pi_n_xsec_net import TrainConfig, PiPlusNElectroproductionRegressor

def _main():
    cfg = TrainConfig()
    cfg.data_path = '/Users/golda/Documents/Study/pi_n_xsec/data/df_maid_as_exp.csv'
    cfg.run_name+='_MAID'
    cfg.phi_to_rad=False

    PiPlusNElectroproductionRegressor.LABEL_COLUMN = 'dsigma_dOmega_maid'
    PiPlusNElectroproductionRegressor.DATA_COLUMNS = ["id","Ebeam", "W", "Q2", "cos_theta", "phi", "dsigma_dOmega",
                                                      "error", "weight", "dsigma_dOmega_maid"]
    model = PiPlusNElectroproductionRegressor(cfg)
    metrics = model.fit()
    print(f"Done. Test MAE = {metrics['mae']:.6f}, Test MSE = {metrics['mse']:.6f}")

if __name__ == "__main__":
    _main()
