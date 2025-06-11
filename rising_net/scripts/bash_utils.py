# -*- coding: utf-8 -*-
import numpy as np

NG = 11
NP = 2000
NPP = 100
NR = 3
NRF = 10
NW = 4  # 7
NL = 3

NEST_LESIONS = [
    # {"DCN_EXC": {'conn_weights.mossy_to_dcn_glut_large': 2*0.554,
    #              'conn_weights.purkinje_to_dcn_glut_large': 0.297/2}},
    # {"DCN_INH": {'conn_weights.mossy_to_dcn_glut_large': 0.554/2,
    #              'conn_weights.purkinje_to_dcn_glut_large': 2*0.297}},
    # {"DCN_INCR": {'conn_weights.mossy_to_dcn_glut_large': 2*0.554,
    #               'conn_weights.purkinje_to_dcn_glut_large': 2*0.297}},
    # {"DCN_DECR": {'conn_weights.mossy_to_dcn_glut_large': 0.554/2,
    #               'conn_weights.purkinje_to_dcn_glut_large': 0.297/2}},
    # {"MOSSY_DCN_INCR_GLOM_DECR": {'conn_weights.mossy_to_dcn_glut_large': 2*0.554,
    #                               'conn_weights.mossy_to_glomerulus': 0.297/2}},
    # {"MOSSY_DCN_DECR_GLOM_INCR": {'conn_weights.mossy_to_dcn_glut_large': 0.554/2,
    #                               'conn_weights.mossy_to_glomerulus': 2*0.297}},
    # {"PURK_TO_DCN_SLOW_02": {"neuron_param.dcn_cell_glut_large.tau_syn2": 2*0.7}}  # ,
    # {"PURK_TO_DCN_SLOW_10": {"neuron_param.dcn_cell_glut_large.tau_syn2": 10*0.7}}
    {"INP_TO_DCN_SLOW": {"neuron_param.dcn_cell_glut_large.tau_syn1": 6.5*1.0,
                         "neuron_param.dcn_cell_glut_large.tau_syn2": 10*0.7}},
    {"INH_TO_DCN_DELAY": {'conn_delays.purkinje_to_dcn_glut_large': 4.0*10}},
    {"INP_TO_GRC": {'conn_weights.glomerulus_to_granule': 0.232/2,
                    'conn_weights.golgi_to_granule': 0.148/2}},
]


def jobarr_id_to_task_ids(args):
    return np.unravel_index(args[0], args[1:], order='C')


def simulate_rest_ids_args(jobarr_id, Ngs=NG, Nreps=NR):
    output = print("--iG %d --iR %d" % jobarr_id_to_task_ids([int(jobarr_id), int(Ngs), int(Nreps)]))
    return output


def simulate_rest_train_ids_args(jobarr_id, Nps=NP, Ngs=NG, Nreps=NR):
    output = print("--iP %d --iG %d --iR %d" % jobarr_id_to_task_ids([int(jobarr_id), int(Nps), int(Ngs), int(Nreps)]))
    return output


def simulate_rest_ppc_ids_args(jobarr_id, Nps=NPP, Ngs=NG, Nreps=NR):
    iP, iG, iR = jobarr_id_to_task_ids([int(jobarr_id), int(Nps), int(Ngs), int(Nreps)])
    output = print("--iP %d --iG %d --iR %d" % (iP, iG, iR))
    return output


def simulate_rest_mapmean_ids_args(jobarr_id, Ngs=NG, Nreps=NR):
    iG, iR = jobarr_id_to_task_ids([int(jobarr_id), int(Ngs), int(Nreps)])
    output = print("--iG %d --iR %d" % (iG, iR))
    return output


def simulate_task_train_ids_args(jobarr_id, Nps=NP, Ngs=NG):
    iG, iP = jobarr_id_to_task_ids([int(jobarr_id), int(Ngs), int(Nps)])
    iG = iG + 4
    output = print("--iG %d --iP %d" % (iG, iP))
    return output


def simulate_task_ppc_ids_args(jobarr_id, Nps=NPP, Ngs=NG):
    iG, iP = jobarr_id_to_task_ids([int(jobarr_id), int(Ngs), int(Nps)])
    iP = 10 * iP
    output = print("--iG %d --iP %d" % (iG, iP))
    return output


def simulate_task_ppc_allruns_ids_args(jobarr_id, Nps=NPP, Ngs=NG):
    iG, iP = jobarr_id_to_task_ids([int(jobarr_id), int(Ngs), int(Nps)])
    iP = 10*iP
    output = print("--iG %d --iP %d" % (iG, iP))
    return output


def fit_rest(jobarr_id):
    from rising_net.scripts.run_fit_plot import GSTR, RESSTR, NSDSTR
    from rising_net.scripts.rest_run_fit_plot import get_config, infer_nRuns_for_iG

    # Fitting:
    iG = int(jobarr_id)
    config = get_config(iG=iG, FUNCMODE="FIT", BASENAME="FIT_REST", verbosity=2)[0]

    return infer_nRuns_for_iG(iG, train_params_samples=None,
                              round=1, prior=None, inference=None, proposal=None,
                              sim_res=None, sim_res_path=None,
                              target=None, ground_truth=None,
                              config=config, igstr=GSTR, folderstr=NSDSTR, resstr=RESSTR,
                              label="", n_samples_per_run=None,
                              save_samples=True, plot_flag=True, verbosity=2)


def fit_task(jobarr_id):
    from rising_net.scripts.run_fit_plot import GSTR, RESSTR, NSDSTR
    from rising_net.scripts.task_run_fit_plot import get_config, infer_nRuns_for_task, \
        get_sim_res_COHM1S1diffratio_gamma, target_COHM1S1diffratio_gamma_fun  # , \
        # get_sim_res_COHM1S1diffratio_allbands, target_COHM1S1diffratio_allbands_fun


    # Fitting:
    iG = int(jobarr_id)
    print("iG1 = %d" % iG)
    config = get_config(iG=iG, FUNCMODE="FIT", BASENAME="FIT_TASKn4", verbosity=2)[0]

    if config.COHERENCE_FISHER_Z_TRANSFORM:
        measure_labels = [
                          # "M1S1R_ThCOHFisherZdiffratio", "M1S1L_ThCOHFisherZdiffratio",
                          # "M1S1R_BtCOHFisherZdiffratio", "M1S1L_BtCOHFisherZdiffratio",
                          "M1S1R_GmCOHFisherZdiffratio", "M1S1L_GmCOHFisherZdiffratio"]
    else:
        measure_labels = [
                          # "M1S1R_ThCOHdiffratio", "M1S1L_ThCOHdiffratio",
                          # "M1S1R_BtCOHdiffratio", "M1S1L_BtCOHdiffratio",
                          "M1S1R_GmCOHdiffratio", "M1S1L_GmCOHdiffratio"]

    return infer_nRuns_for_task(iG=iG, train_params_samples=None,
                                round=0, prior=None, inference=None, proposal=None,
                                sim_res=None, sim_res_path=None,
                                sim_res_fun=get_sim_res_COHM1S1diffratio_gamma,  # get_sim_res_COHM1S1diffratio_allbands
                                target=None,
                                target_fun=target_COHM1S1diffratio_gamma_fun,  # target_COHM1S1diffratio_allbands_fun
                                ground_truth=None,
                                config=config, folderstr=NSDSTR, resstr=RESSTR,
                                label="", n_samples_per_run=None, measure_labels=measure_labels,
                                save_samples=True, plot_flag=True, verbosity=2)


def simulate_cosim_wTVBtoNEST_fit(jobarr_id, Nws=NW, Ngs=NG, Nreps=NR):
    from rising_net.scripts.task_run_fit_plot import sim_run_plot

    iG, iW, iR = jobarr_id_to_task_ids([int(jobarr_id), int(Ngs), int(Nws), int(Nreps)])
    iG += 4
    w_TVB_to_NEST = 35 + 5 * iW
    force_output_folder = "wTVBtoNESTfit/iG_%02d/w%03d/nsd_%d" % (iG, int(10*w_TVB_to_NEST), iR)
    return sim_run_plot(iG=iG, iP=None, iR=None,
                        FUNCMODE="MEANSIM",
                        label="",
                        config=None, REST_or_TASK="TASK",
                        force_output_folder=force_output_folder,
                        fitlabel="allsamples",
                        REST_BASENAME="FIT_REST",
                        restfitlabel="allsamples",
                        MODE="COSIM",
                        BASENAME="FIT_TASK",
                        SIMULATION_LENGTH=2 ** 13 + 1.0,
                        w_TVB_to_NEST=w_TVB_to_NEST,
                        # NOISE=1e-6,
                        verbosity=2
                     )


def simulate_cosim_wNESTtoTVB_fit(jobarr_id, Ngs=NG, Nreps=NR):

    from rising_net.scripts.task_run_fit_plot import sim_run_plot

    jobarr_id = int(jobarr_id)
    iG, iR = jobarr_id_to_task_ids([jobarr_id // 2, int(Ngs), int(Nreps)])
    iG += 4

    if np.mod(jobarr_id, 2) == 0:
        REST_or_TASK = "TASK"
        print("\n" + "-"*50 + "\nSIMULATING COSIM %s for G=%d, iR=%d!\n" % (REST_or_TASK, iG, iR) + "-"*50 + "\n")
        force_output_folder = "wNESTtoTVBfit/COSIM_%s/iG_%02d/nsd_%d" % (REST_or_TASK, iG, iR)
        return sim_run_plot(iG=iG, iP=None, iR=None,
                            FUNCMODE="MEANSIM",
                            label="",
                            config=None, REST_or_TASK=REST_or_TASK,
                            force_output_folder=force_output_folder,
                            fitlabel="allsamples",
                            REST_BASENAME="FIT_REST",
                            restfitlabel="allsamples",
                            MODE="COSIM",
                            BASENAME="FIT_TASK",
                            SIMULATION_LENGTH=2 ** 13 + 1.0,
                            # NOISE=1e-6,
                            verbosity=2
                            )
    else:
        REST_or_TASK = "REST"
        print("\n" + "-" * 50 + "\nSIMULATING COSIM %s for G=%d, iR=%d!\n" % (REST_or_TASK, iG, iR) + "-" * 50 + "\n")
        force_output_folder = "wNESTtoTVBfit/COSIM_%s/iG_%02d/nsd_%d" % (REST_or_TASK, iG, iR)
        return sim_run_plot(iG=iG, iP=None, iR=None,
                            FUNCMODE="MEANSIM",
                            label="",
                            config=None, REST_or_TASK=REST_or_TASK,
                            force_output_folder=force_output_folder,
                            fitlabel="allsamples",
                            REST_BASENAME="FIT_REST",
                            restfitlabel="allsamples",
                            MODE="COSIM",
                            BASENAME="FIT_TASK",
                            SIMULATION_LENGTH=2 ** 13 + 1.0,
                            # NOISE=1e-6,
                            verbosity=2
                            )


def simulate_tvb_CEREBON_OFF(jobarr_id, Ngs=NG, Nreps=NRF):

    from rising_net.scripts.task_run_fit_plot import sim_run_plot

    jobarr_id = int(jobarr_id)
    iG, iR = jobarr_id_to_task_ids([jobarr_id // 2, int(Ngs), int(Nreps)])
    iG += 4

    if np.mod(jobarr_id, 2) == 0:
        print("\n" + "-"*50 + "\nSIMULATING TVB CEREBON for G=%d, iR=%d!\n" % (iG, iR) + "-"*50 + "\n")
        force_output_folder = "TVB_CEREBON_OFF/iG_%02d/nsd_%d/TVB/" % (iG, iR)
        return sim_run_plot(iG=iG, iP=None, iR=iR,
                            FUNCMODE="MEANSIM",  # "BESTSIM2",  # "MEANSIM",
                            label="",
                            config=None, REST_or_TASK="TASK",
                            force_output_folder=force_output_folder,
                            REST_BASENAME="FIT_REST",
                            restfitlabel="allsamples",
                            MODE="TVB",
                            BASENAME="FIT_TASK",
                            fitlabel="allsamples",
                            SIMULATION_LENGTH=2 ** 13 + 1.0,
                            # NOISE=1e-6,
                            verbosity=2
                            )
    else:
        print("\n\n\n\n" + "-" * 50 + "\n\n" +
              "\nSIMULATING TVB _CEREBOFF for G=%d, iR=%d!\n" % (iG, iR) + "-" * 50 + "\n")
        force_output_folder = "TVB_CEREBON_OFF/iG_%02d/nsd_%d/TVB_CEREBOFF" % (iG, iR)
        return sim_run_plot(iG=iG, iP=None, iR=iR,
                            FUNCMODE="MEANSIM",  # "BESTSIM2",  # "MEANSIM",
                            label="",
                            config=None, REST_or_TASK="TASK",
                            force_output_folder=force_output_folder,
                            REST_BASENAME="FIT_REST",
                            restfitlabel="allsamples",
                            MODE="TVB_CEREBOFF",
                            BASENAME="FIT_TASK",
                            fitlabel="allsamples",
                            SIMULATION_LENGTH=2 ** 13 + 1.0,
                            # NOISE=1e-6,
                            verbosity=2
                            )


def simulate_cosim_CEREBON_OFF(jobarr_id, Ngs=NG, Nreps=NRF):

    from rising_net.scripts.task_run_fit_plot import sim_run_plot

    jobarr_id = int(jobarr_id)
    iG, iR = jobarr_id_to_task_ids([jobarr_id // 2, int(Ngs), int(Nreps)])
    iG += 4

    if np.mod(jobarr_id, 2) == 0:
        print("\n" + "-"*50 + "\nSIMULATING COSIM CEREBON for G=%d, iR=%d!\n" % (iG, iR) + "-"*50 + "\n")
        force_output_folder = "COSIM_CEREBON_OFF/iG_%02d/nsd_%d/COSIM/" % (iG, iR)
        return sim_run_plot(iG=iG, iP=None, iR=iR,
                            FUNCMODE="MEANSIM",
                            label="",
                            config=None, REST_or_TASK="TASK",
                            force_output_folder=force_output_folder,
                            REST_BASENAME="FIT_REST",
                            restfitlabel="allsamples",
                            MODE="COSIM",
                            BASENAME="FIT_TASK",
                            fitlabel="allsamples",
                            SIMULATION_LENGTH=2 ** 13 + 1.0,
                            # NOISE=1e-4,
                            verbosity=2
                            )
    else:
        print("\n\n\n\n" + "-" * 50 + "\n\n" +
              "\nSIMULATING COSIM _CEREBOFF for G=%d, iR=%d!\n" % (iG, iR) + "-" * 50 + "\n")
        force_output_folder = "COSIM_CEREBON_OFF/iG_%02d/nsd_%d/COSIM_CEREBOFF" % (iG, iR)
        return sim_run_plot(iG=iG, iP=None, iR=iR,
                            FUNCMODE="MEANSIM",
                            label="",
                            config=None, REST_or_TASK="TASK",
                            force_output_folder=force_output_folder,
                            REST_BASENAME="FIT_REST",
                            restfitlabel="allsamples",
                            MODE="COSIM_CEREBOFF",
                            BASENAME="FIT_TASK",
                            fitlabel="allsamples",
                            SIMULATION_LENGTH=2 ** 13 + 1.0,
                            # NOISE=1e-4,
                            verbosity=2
                            )


def simulate_cosim_nest_lesion(jobarr_id, Nls=NL, Nreps=NRF):

    from rising_net.scripts.task_run_fit_plot import sim_run_plot

    iL, iR = jobarr_id_to_task_ids([int(jobarr_id), int(Nls), int(Nreps)])
    lesion = NEST_LESIONS[iL]
    lname = list(lesion.keys())[0]
    lval = list(lesion.values())[0]
    print("\n" + "-"*50 + "\nSIMULATING COSIM NEST LESION %d, iR=%d!\n" % (iL, iR) + "-"*50 + "\n")
    print("Lesion %s:\n%s\n" % (lname, str(lval)))
    MODE = "COSIM_%s" % lname
    force_output_folder = "COSIM_NEST_LESIONnorm/nsd_%d/%s/" % (iR, MODE)
    return sim_run_plot(iG=6, iP=None, iR=iR,
                        FUNCMODE="MEANSIM",
                        label="",
                        config=None, REST_or_TASK="TASK",
                        force_output_folder=force_output_folder,
                        REST_BASENAME="FIT_REST",
                        restfitlabel="allsamples",
                        MODE=MODE,
                        BASENAME="FIT_TASK",
                        fitlabel="allsamples",
                        SIMULATION_LENGTH=2 ** 13 + 1.0,
                        # NOISE=1e-4,
                        verbosity=2,
                        nest_lesions=lval
                        )


def multiply(x, y):
    output = print(x*y)
    return output


def condition(x, th=100.0):
    if x <= th:
        output = 1
    else:
        output = 0
    print(output)
    return output


if __name__ == '__main__':
    import sys

    if sys.argv[1] == "multiply":
        multiply(float(sys.argv[2]), float(sys.argv[3]))
    elif sys.argv[1] == "condition":
        x = float(sys.argv[2])
        if len(sys.argv) > 3:
            condition(x, float(sys.argv[3]))
        else:
            condition(x)
    elif sys.argv[1] == "fit_rest":
        fit_rest(*sys.argv[2:])
    elif sys.argv[1] == "simulate_rest_ids_args":
        simulate_rest_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "simulate_rest_train_ids_args":
        simulate_rest_train_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "simulate_rest_ppc_ids_args":
        simulate_rest_ppc_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "simulate_rest_mapmean_ids_args":
        simulate_rest_mapmean_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "fit_task":
        fit_task(*sys.argv[2:])
    elif sys.argv[1] == "simulate_task_train_ids_args":
        simulate_task_train_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "simulate_task_ppc_ids_args":
        simulate_task_ppc_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "simulate_task_ppc_allruns_ids_args":
        simulate_task_ppc_allruns_ids_args(*sys.argv[2:])
    elif sys.argv[1] == "simulate_cosim_wTVBtoNEST_fit":
        simulate_cosim_wTVBtoNEST_fit(*sys.argv[2:])
    elif sys.argv[1] == "simulate_cosim_wNESTtoTVB_fit":
        simulate_cosim_wNESTtoTVB_fit(*sys.argv[2:])
    elif sys.argv[1] == "simulate_tvb_CEREBON_OFF":
        simulate_tvb_CEREBON_OFF(*sys.argv[2:])
    elif sys.argv[1] == "simulate_cosim_CEREBON_OFF":
        simulate_cosim_CEREBON_OFF(*sys.argv[2:])
    elif sys.argv[1] == "simulate_cosim_nest_lesion":
        simulate_cosim_nest_lesion(*sys.argv[2:])
