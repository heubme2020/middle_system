import time

import numpy as np
import pandas as pd
from chinese_calendar import is_workday
import tools
import pandas as pd
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
import datetime
import tushare as ts
import cv2
import os
import random

ts.set_token('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')
pro = ts.pro_api()
pro = ts.pro_api('d7dc8dcedbac88a7179f9100c2b2d40b8a322dce8da6c080dc8d1c90')

##参数列表
##存储利润表的参数列表
income_param_list = ['basic_eps', 'diluted_eps', 'total_revenue', 'revenue', 'int_income', 'prem_earned',
                     'comm_income', 'n_commis_income',
                     'n_oth_income', 'n_oth_b_income', 'prem_income', 'out_prem', 'une_prem_reser', 'reins_income',
                     'n_sec_tb_income', 'n_sec_uw_income',
                     'n_asset_mg_income', 'oth_b_income', 'fv_value_chg_gain', 'invest_income', 'ass_invest_income',
                     'forex_gain', 'total_cogs', 'oper_cost',
                     'int_exp', 'comm_exp', 'biz_tax_surchg', 'sell_exp', 'admin_exp', 'fin_exp',
                     'assets_impair_loss', 'prem_refund',
                     'compens_payout', 'reser_insur_liab', 'div_payt', 'reins_exp', 'oper_exp',
                     'compens_payout_refu', 'insur_reser_refu', 'reins_cost_refund',
                     'other_bus_cost', 'operate_profit', 'non_oper_income', 'non_oper_exp', 'nca_disploss',
                     'total_profit', 'income_tax', 'n_income',
                     'n_income_attr_p', 'minority_gain', 'oth_compr_income', 't_compr_income', 'compr_inc_attr_p',
                     'compr_inc_attr_m_s', 'ebit', 'ebitda',
                     'undist_profit', 'distable_profit', 'rd_exp', 'fin_exp_int_exp', 'fin_exp_int_inc',
                     'transfer_surplus_rese', 'transfer_housing_imprest', 'transfer_oth',
                     'adj_lossgain', 'withdra_legal_surplus', 'withdra_legal_pubfund', 'withdra_biz_devfund',
                     'withdra_rese_fund', 'withdra_oth_ersu', 'workers_welfare',
                     'distr_profit_shrhder', 'prfshare_payable_dvd', 'comshare_payable_dvd', 'capit_comstock_div']
##存储利润表参数的缩放倍数
income_zoom_list = [10000.0, 10000.0, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                    1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                    1.0 / 10000,
                    1.0 / 10000, 1.0 / 10000, 1.0, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                    1.0 / 10000,
                    1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                    1.0 / 10000,
                    1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 100000, 1.0 / 10000, 1.0 / 10000,
                    1.0 / 10000,
                    1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                    1.0 / 10000,
                    1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                    1.0 / 10000,
                    1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                    1.0 / 10000,
                    1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                    1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000]
##存储负债表的参数列表
balancesheet_param_list = ['total_share', 'cap_rese', 'undistr_porfit', 'surplus_rese', 'special_rese', 'money_cap',
                           'trad_asset', 'notes_receiv',
                           'accounts_receiv', 'oth_receiv', 'prepayment', 'div_receiv', 'int_receiv', 'inventories',
                           'nca_within_1y', 'sett_rsrv',
                           'loanto_oth_bank_fi', 'premium_receiv', 'reinsur_receiv', 'pur_resale_fa', 'oth_cur_assets',
                           'total_cur_assets', 'fa_avail_for_sale', 'htm_invest',
                           'lt_eqt_invest', 'invest_real_estate', 'time_deposits', 'oth_assets', 'lt_rec', 'fix_assets',
                           'cip', 'const_materials',
                           'fixed_assets_disp', 'intan_assets', 'r_and_d', 'goodwill', 'lt_amor_exp',
                           'defer_tax_assets', 'decr_in_disbur', 'oth_nca',
                           'total_nca', 'cash_reser_cb', 'depos_in_oth_bfi', 'prec_metals', 'deriv_assets',
                           'rr_reins_une_prem', 'refund_depos', 'ph_pledge_loans',
                           'indep_acct_assets', 'client_depos', 'transac_seat_fee', 'invest_as_receiv', 'total_assets',
                           'lt_borr', 'st_borr', 'cb_borr',
                           'depos_ib_deposits', 'loan_oth_bank', 'trading_fl', 'notes_payable', 'acct_payable',
                           'adv_receipts', 'sold_for_repur_fa', 'comm_payable',
                           'payroll_payable', 'taxes_payable', 'int_payable', 'div_payable', 'oth_payable', 'acc_exp',
                           'deferred_inc', 'st_bonds_payable',
                           'rsrv_insur_cont', 'acting_trading_sec', 'acting_uw_sec', 'non_cur_liab_due_1y',
                           'oth_cur_liab', 'total_cur_liab', 'bond_payable', 'lt_payable',
                           'specific_payables', 'estimated_liab', 'defer_tax_liab', 'defer_inc_non_cur_liab', 'oth_ncl',
                           'total_ncl', 'depos_oth_bfi', 'deriv_liab',
                           'depos', 'oth_liab', 'prem_receiv_adva', 'depos_received', 'ph_invest', 'reser_une_prem',
                           'pledge_borr',
                           'indem_payable', 'policy_div_payable', 'total_liab', 'treasury_share', 'forex_differ',
                           'invest_loss_unconf', 'minority_int', 'total_hldr_eqy_exc_min_int',
                           'total_hldr_eqy_inc_min_int', 'total_liab_hldr_eqy', 'lt_payroll_payable', 'oth_comp_income',
                           'oth_eqt_tools', 'oth_eqt_tools_p_shr', 'lending_funds', 'acc_receivable',
                           'st_fin_payable', 'payables', 'hfs_assets', 'hfs_sales', 'cost_fin_assets',
                           'fair_value_fin_assets', 'cip_total', 'oth_pay_total',
                           'long_pay_total', 'debt_invest', 'oth_debt_invest', 'contract_assets', 'contract_liab',
                           'accounts_receiv_bill', 'accounts_pay', 'oth_rcv_total', 'fix_assets_total']
##存储负债表参数的缩放倍数
balancesheet_zoom_list = [1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000,
                          1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                          1.0 / 10000]

##存储现金流量表的参数列表
cashflow_param_list = ['net_profit', 'finan_exp', 'c_fr_sale_sg', 'recp_tax_rends', 'n_depos_incr_fi',
                       'n_incr_loans_cb', 'n_inc_borr_oth_fi', 'prem_fr_orig_contr',
                       'n_incr_insured_dep', 'n_reinsur_prem', 'n_incr_disp_tfa', 'ifc_cash_incr', 'n_incr_disp_faas',
                       'n_incr_loans_oth_bank', 'n_cap_incr_repur', 'c_fr_oth_operate_a',
                       'c_inf_fr_operate_a', 'c_paid_goods_s', 'c_paid_to_for_empl', 'c_paid_for_taxes',
                       'n_incr_clt_loan_adv', 'n_incr_dep_cbob', 'c_pay_claims_orig_inco', 'pay_handling_chrg',
                       'pay_comm_insur_plcy', 'oth_cash_pay_oper_act', 'st_cash_out_act', 'n_cashflow_act',
                       'oth_recp_ral_inv_act', 'c_disp_withdrwl_invest', 'c_recp_return_invest', 'n_recp_disp_fiolta',
                       'n_recp_disp_sobu', 'stot_inflows_inv_act', 'c_pay_acq_const_fiolta', 'c_paid_invest',
                       'n_disp_subs_oth_biz', 'oth_pay_ral_inv_act', 'n_incr_pledge_loan', 'stot_out_inv_act',
                       'n_cashflow_inv_act', 'c_recp_borrow', 'proc_issue_bonds', 'oth_cash_recp_ral_fnc_act',
                       'stot_cash_in_fnc_act', 'free_cashflow', 'c_prepay_amt_borr', 'c_pay_dist_dpcp_int_exp',
                       'incl_dvd_profit_paid_sc_ms', 'oth_cashpay_ral_fnc_act', 'stot_cashout_fnc_act',
                       'n_cash_flows_fnc_act', 'eff_fx_flu_cash', 'n_incr_cash_cash_equ', 'c_cash_equ_beg_period',
                       'c_cash_equ_end_period',
                       'c_recp_cap_contrib', 'incl_cash_rec_saims', 'uncon_invest_loss', 'prov_depr_assets',
                       'depr_fa_coga_dpba', 'amort_intang_assets', 'lt_amort_deferred_exp', 'decr_deferred_exp',
                       'incr_acc_exp', 'loss_disp_fiolta', 'loss_scr_fa', 'loss_fv_chg', 'invest_loss',
                       'decr_def_inc_tax_assets', 'incr_def_inc_tax_liab', 'incr_def_inc_tax_liab',
                       'decr_inventories', 'decr_oper_payable', 'incr_oper_payable', 'others',
                       'im_net_cashflow_oper_act', 'conv_debt_into_cap', 'conv_copbonds_due_within_1y', 'fa_fnc_leases',
                       'im_n_incr_cash_equ', 'net_dism_capital_add', 'net_cash_rece_sec', 'credit_impa_loss',
                       'use_right_asset_dep', 'oth_loss_asset', 'end_bal_cash', 'beg_bal_cash',
                       'end_bal_cash_equ', 'beg_bal_cash_equ', 'update_flag']
##存储现金流量表参数的缩放倍数
cashflow_zoom_list = [1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                      1.0 / 10000,
                      1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                      1.0 / 10000,
                      1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                      1.0 / 10000,
                      1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                      1.0 / 10000,
                      1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                      1.0 / 10000,
                      1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                      1.0 / 10000,
                      1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                      1.0 / 10000,
                      1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                      1.0 / 10000,
                      1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                      1.0 / 10000,
                      1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                      1.0 / 10000,
                      1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                      1.0 / 10000,
                      1.0 / 10000, 1.0 / 10000, 1.0]
##存储财务指标数据的参数列表
fina_indicator_param_list = ['eps', 'dt_eps', 'total_revenue_ps', 'revenue_ps', 'capital_rese_ps', 'surplus_rese_ps',
                             'undist_profit_ps', 'extra_item',
                             'profit_dedt', 'gross_margin', 'current_ratio', 'quick_ratio', 'cash_ratio', 'ar_turn',
                             'ca_turn', 'fa_turn',
                             'assets_turn', 'op_income', 'ebit', 'ebitda', 'fcff', 'fcfe', 'current_exint',
                             'noncurrent_exint',
                             'interestdebt', 'netdebt', 'tangible_asset', 'working_capital', 'networking_capital',
                             'invest_capital', 'retained_earnings', 'diluted2_eps',
                             'bps', 'ocfps', 'retainedps', 'cfps', 'ebit_ps', 'fcff_ps', 'fcfe_ps', 'netprofit_margin',
                             'grossprofit_margin', 'cogs_of_sales', 'expense_of_sales', 'profit_to_gr', 'saleexp_to_gr',
                             'adminexp_of_gr', 'finaexp_of_gr', 'impai_ttm',
                             'gc_of_gr', 'op_of_gr', 'ebit_of_gr', 'roe', 'roe_waa', 'roe_dt', 'roa', 'npta',
                             'roic', 'roe_yearly', 'roa2_yearly', 'assets_to_eqt', 'dp_assets_to_eqt', 'ca_to_assets',
                             'nca_to_assets', 'tbassets_to_totalassets',
                             'int_to_talcap', 'eqt_to_talcapital', 'currentdebt_to_debt', 'longdeb_to_debt',
                             'ocf_to_shortdebt', 'debt_to_eqt', 'eqt_to_debt', 'eqt_to_interestdebt',
                             'tangibleasset_to_debt', 'tangasset_to_intdebt', 'tangibleasset_to_netdebt', 'ocf_to_debt',
                             'turn_days', 'roa_yearly', 'roa_dp', 'fixed_assets',
                             'profit_to_op', 'q_saleexp_to_gr', 'q_gc_to_gr', 'q_roe', 'q_dt_roe', 'q_npta',
                             'q_ocf_to_sales', 'basic_eps_yoy',
                             'dt_eps_yoy', 'cfps_yoy', 'op_yoy', 'ebt_yoy', 'netprofit_yoy', 'dt_netprofit_yoy',
                             'ocf_yoy', 'roe_yoy',
                             'bps_yoy', 'assets_yoy', 'eqt_yoy', 'tr_yoy', 'or_yoy', 'q_sales_yoy', 'q_op_qoq',
                             'equity_yoy']
##存储财务指标数据的缩放倍数
fina_indicator_zoom_list = [10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 1.0 / 10000,
                            1.0 / 10000, 1.0 / 10000, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                            10000.0, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                            1.0 / 10000,
                            1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000, 1.0 / 10000,
                            10000.0,
                            10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                            10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                            10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                            10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                            10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                            10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                            10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                            10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
                            10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0]
#获取行情数据
def get_trade_date_list(trade_date, trade_date_num):
    trade_date_list = []
    today_date = tools.get_date_today()
    if trade_date > today_date:
        trade_date = today_date
    start_date = tools.get_delta_date(trade_date, -3*trade_date_num)
    trade_cal = pro.trade_cal(start_date=start_date, end_date=trade_date)
    trade_cal = trade_cal.drop(trade_cal[trade_cal['is_open']==0].index)
    trade_cal = trade_cal.sort_values(by='cal_date', ascending=False)
    for i in range(trade_date_num):
        trade_date_list.append(trade_cal['cal_date'].iloc[i])
    return trade_date_list

def get_trade_date_list_period(start_date, end_date):
    trade_date_list = []
    trade_cal = pro.trade_cal(start_date=start_date, end_date=end_date)
    trade_cal = trade_cal.drop(trade_cal[trade_cal['is_open']==0].index)
    trade_cal = trade_cal.sort_values(by='cal_date', ascending=False)
    for i in range(trade_cal.shape[0]):
        trade_date_list.append(trade_cal['cal_date'].iloc[i])
    return trade_date_list

def get_n_half_hour_data(ts_code, trade_date, n=32):
    trade_date_list = get_trade_date_list(trade_date, n+7)
    start_date = trade_date_list[-1] + ' 09:00:00'
    start_hour = str(datetime.datetime.strptime(start_date, '%Y%m%d %H:%M:%S'))
    end_date = trade_date_list[0] + ' 17:00:00'
    end_hour = str(datetime.datetime.strptime(end_date, '%Y%m%d %H:%M:%S'))
    half_hour_data = ts.pro_bar(ts_code = ts_code, freq='30min', start_date = start_hour, end_date = end_hour)
    half_hour_data = half_hour_data.head(int(n*9))
    half_hour_data = half_hour_data.fillna(-1)
    return half_hour_data


def get_n_daily_data(ts_code, trade_date, n=64):
    trade_date_list = get_trade_date_list(trade_date, n+3)
    daily_data = pro.daily(ts_code = ts_code, start_date = trade_date_list[-1], end_date = trade_date_list[0])
    daily_data = daily_data.head(n)
    daily_data = daily_data.fillna(-1)
    return daily_data

def get_n_daily_basic_data(ts_code, trade_date, n=64):
    trade_date_list = get_trade_date_list(trade_date, n+3)
    daily_basic_data = pro.daily_basic(ts_code = ts_code, start_date = trade_date_list[-1], end_date = trade_date_list[0])
    daily_basic_data = daily_basic_data.head(n)
    daily_basic_data = daily_basic_data.fillna(-1)
    return daily_basic_data

def get_n_moneyflow_data(ts_code, trade_date, n=64):
    trade_date_list = get_trade_date_list(trade_date, n+3)
    moneyflow_data = pro.moneyflow(ts_code = ts_code, start_date = trade_date_list[-1], end_date = trade_date_list[0])
    moneyflow_data = moneyflow_data.head(n)
    moneyflow_data = moneyflow_data.fillna(-1)
    return moneyflow_data

def get_n_bak_daily_data(ts_code, trade_date, n=64):
    trade_date_list = get_trade_date_list(trade_date, n+3)
    bak_daily_data = pro.bak_daily(ts_code = ts_code, start_date = trade_date_list[-1], end_date = trade_date_list[0])
    bak_daily_data = bak_daily_data.head(n)
    bak_daily_data = bak_daily_data.fillna(-1)
    return bak_daily_data


def get_n_weekly_data(ts_code, trade_date, n=56):
    trade_date_list = get_trade_date_list(trade_date, n*7)
    weekly_data = pro.weekly(ts_code = ts_code, start_date = trade_date_list[-1], end_date = trade_date_list[0])
    weekly_data = weekly_data.head(n)
    weekly_data = weekly_data.fillna(-1)
    return weekly_data

def get_n_monthly_data(ts_code, trade_date, n=40):
    trade_date_list = get_trade_date_list(trade_date, n*22)
    monthly_data = pro.monthly(ts_code = ts_code, start_date = trade_date_list[-1], end_date = trade_date_list[0])
    monthly_data = monthly_data.head(n)
    monthly_data = monthly_data.fillna(-1)
    return monthly_data

#获取财务数据
def get_n_income_data(ts_code, trade_date, n=16):
    trade_date_list = get_trade_date_list(trade_date, n*100)
    income_data = pro.income(ts_code = ts_code, start_date = trade_date_list[-1], end_date = trade_date_list[0])
    income_data = income_data.drop_duplicates(subset=['end_date'], keep='first')
    income_data = income_data.reset_index(drop=True)
    income_data = income_data.head(n)
    income_data = income_data.fillna(-1)
    return income_data

def get_n_balancesheet_data(ts_code, trade_date, n=16):
    trade_date_list = get_trade_date_list(trade_date, n*100)
    balancesheet_data = pro.balancesheet(ts_code = ts_code, start_date = trade_date_list[-1], end_date = trade_date_list[0])
    balancesheet_data = balancesheet_data.drop_duplicates(subset=['end_date'], keep='first')
    balancesheet_data = balancesheet_data.reset_index(drop=True)
    balancesheet_data = balancesheet_data.head(n)
    balancesheet_data = balancesheet_data.fillna(-1)
    return balancesheet_data

def get_n_cashflow_data(ts_code, trade_date, n=16):
    trade_date_list = get_trade_date_list(trade_date, n*100)
    cashflow_data = pro.cashflow(ts_code = ts_code, start_date = trade_date_list[-1], end_date = trade_date_list[0])
    cashflow_data = cashflow_data.drop_duplicates(subset=['end_date'], keep='first')
    cashflow_data = cashflow_data.reset_index(drop=True)
    cashflow_data = cashflow_data.head(n)
    cashflow_data = cashflow_data.fillna(-1)
    return cashflow_data

def get_n_fina_indicator_data(ts_code, trade_date, n=16):
    trade_date_list = get_trade_date_list(trade_date, n*100)
    fina_indicator_data = pro.fina_indicator(ts_code = ts_code, start_date = trade_date_list[-1], end_date = trade_date_list[0])
    fina_indicator_data = fina_indicator_data.drop_duplicates(subset=['end_date'], keep='first')
    fina_indicator_data = fina_indicator_data.reset_index(drop=True)
    fina_indicator_data = fina_indicator_data.head(n)
    fina_indicator_data = fina_indicator_data.fillna(-1)
    return fina_indicator_data

#将处理好的整数转换为4个点255灰度值，相当于4byte, 输入范围是-2^31：2^31之间，-2147483648：2147483648
def get_int_to_4_255_gray(int_num):
    four_byte = [0, 0, 0, 0]
    complement = [0]*32
    binary_num = bin(int_num)
    for i in range(32):
        if binary_num[-i] == 'b':
            break
        else:
            complement[-i] = binary_num[-i]
    complement[0] = 1
    if int_num < 0:
        for i in range(32):
            complement[i] = 1 - int(complement[i])
    for i in range(4):
        str_byte = ''
        for j in range(8):
            str_byte = str_byte + str(complement[i*8 + j])
        int_byte = int(str_byte, base=2)
        four_byte[i] = int_byte
    return four_byte
#将处理好的整数转换为32个点，这里一个点只代表1个bit, 输入范围是-2^31：2^31之间，-2147483648：2147483648
def get_int_to_32_255_gray(int_num):
    bytes = [0]*32
    binary_num = bin(int_num)
    for i in range(32):
        if binary_num[-i] == 'b':
            break
        else:
            bytes[-i] = binary_num[-i]
    bytes[0] = 1
    if int_num < 0:
        for i in range(32):
            bytes[i] = 1 - int(bytes[i])
    return bytes


def get_first_input_image(ts_code, trade_date):
    ##*******************第一通道财务数据
    input_image = np.zeros([256, 256, 1], dtype="uint8")
    seasons_count = 16
    ##~~~~~~~~~利润表
    income_data = get_n_income_data(ts_code, trade_date, seasons_count)
    if income_data.shape[0] < seasons_count:
        pass
    else:
        for i in range(len(income_param_list)):
            for j in range(seasons_count):
                param = int(float(income_data[income_param_list[i]][j])*income_zoom_list[i])
                param_255 = get_int_to_4_255_gray(param)
                k = int(i/64)
                m = i%64
                input_image[j*2+k][4*m]=int(param_255[0])
                input_image[j*2+k][4*m+1]=int(param_255[1])
                input_image[j*2+k][4*m+2]=int(param_255[2])
                input_image[j*2+k][4*m+3]=int(param_255[3])
    ##~~~~~~~~~资产负债表
    balancesheet_data = get_n_balancesheet_data(ts_code, trade_date, seasons_count)
    if balancesheet_data.shape[0] < seasons_count:
        pass
    else:
        for i in range(len(balancesheet_param_list)):
            for j in range(seasons_count):
                param = int(float(balancesheet_data[balancesheet_param_list[i]][j])*balancesheet_zoom_list[i])
                param_255 = get_int_to_4_255_gray(param)
                k = int(i/64) + seasons_count*2
                m = i%64
                input_image[j*2+k][4*m]=int(param_255[0])
                input_image[j*2+k][4*m+1]=int(param_255[1])
                input_image[j*2+k][4*m+2]=int(param_255[2])
                input_image[j*2+k][4*m+3]=int(param_255[3])
    ##~~~~~~~~~现金流量表
    cashflow_data = get_n_cashflow_data(ts_code, trade_date)
    if cashflow_data.shape[0] < seasons_count:
        pass
    else:
        for i in range(len(cashflow_param_list)):
            for j in range(seasons_count):
                param = int(float(cashflow_data[cashflow_param_list[i]][j])*cashflow_zoom_list[i])
                param_255 = get_int_to_4_255_gray(param)
                k = int(i/64) + seasons_count*4
                m = i%64
                input_image[j*2+k][4*m]=int(param_255[0])
                input_image[j*2+k][4*m+1]=int(param_255[1])
                input_image[j*2+k][4*m+2]=int(param_255[2])
                input_image[j*2+k][4*m+3]=int(param_255[3])

    ##~~~~~~~~~财务指标数据
    fina_indicator_data = get_n_fina_indicator_data(ts_code, trade_date)
    if fina_indicator_data.shape[0] < seasons_count:
        pass
    else:
        for i in range(len(fina_indicator_param_list)):
            for j in range(seasons_count):
                param = int(float(fina_indicator_data[fina_indicator_param_list[i]][j])*fina_indicator_zoom_list[i])
                param_255 = get_int_to_4_255_gray(param)
                k = int(i/64) + seasons_count*6
                m = i%64
                input_image[j*2+k][4*m]=int(param_255[0])
                input_image[j*2+k][4*m+1]=int(param_255[1])
                input_image[j*2+k][4*m+2]=int(param_255[2])
                input_image[j*2+k][4*m+3]=int(param_255[3])
    return input_image
def get_second_input_image(ts_code, trade_date):
    ##*******************第二通道行情数据
    input_image = np.zeros([256, 256, 1], dtype="uint8")
    ##~~~~~~~~~30min行情
    days_count = 32
    half_hour_data = get_n_half_hour_data(ts_code, trade_date, days_count)
    #30min数据的参数列表
    half_hour_param_list = ['open', 'close', 'high', 'low', 'vol', 'amount', 'pre_close']
    ##存储30min数据的缩放倍数
    half_hour_zoom_list = [100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 100.0]

    if half_hour_data.shape[0] < days_count*9:
        pass
    else:
        k = len(half_hour_param_list)
        for i in range(k):
            for j in range(days_count*9):
                param = int(float(half_hour_data[half_hour_param_list[i]][j])*half_hour_zoom_list[i])
                param_255 = get_int_to_4_255_gray(param)
                m = int(j/9)
                n = j%9
                input_image[m][(n*k+i)*4]=int(param_255[0])
                input_image[m][(n*k+i)*4+1]=int(param_255[1])
                input_image[m][(n*k+i)*4+2]=int(param_255[2])
                input_image[m][(n*k+i)*4+3]=int(param_255[3])
    ##~~~~~~~~~日线行情
    days_count = 64
    daily_data = get_n_daily_data(ts_code, trade_date, days_count)
    ##日线数据的参数列表
    daily_param_list = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
    ##存储日线数据的缩放倍数
    daily_zoom_list = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 10000.0, 1.0, 1.0]

    if daily_data.shape[0] < days_count:
        pass
    else:
        k = len(daily_param_list)
        for i in range(k):
            for j in range(days_count):
                param = int(float(daily_data[daily_param_list[i]][j])*daily_zoom_list[i])
                param_255 = get_int_to_4_255_gray(param)
                input_image[j*2+32][4*i]=int(param_255[0])
                input_image[j*2+32][4*i+1]=int(param_255[1])
                input_image[j*2+32][4*i+2]=int(param_255[2])
                input_image[j*2+32][4*i+3]=int(param_255[3])
    ##~~~~~~~~~每日指标行情
    days_count = 64
    daily_basic_data = get_n_daily_basic_data(ts_code, trade_date, days_count)
    ##每日指标数据的参数列表
    daily_basic_param_list =['close', 'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps',
                             'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv']
    ##存储每日指标数据的缩放倍数
    daily_basic_zoom_list = [100.0, 10000.0, 10000.0, 100.0, 10000.0, 10000.0, 10000.0, 10000.0,
                             10000.0, 10000.0, 10000.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    if daily_basic_data.shape[0] < days_count:
        pass
    else:
        k = len(daily_basic_param_list)
        for i in range(k):
            for j in range(days_count):
                param = int(float(daily_basic_data[daily_basic_param_list[i]][j])*daily_basic_zoom_list[i])
                param_255 = get_int_to_4_255_gray(param)
                input_image[j*2+32][4*i+128]=int(param_255[0])
                input_image[j*2+32][4*i+129]=int(param_255[1])
                input_image[j*2+32][4*i+130]=int(param_255[2])
                input_image[j*2+32][4*i+131]=int(param_255[3])

    ##~~~~~~~~~个股资金流向行情
    days_count = 64
    moneyflow_data = get_n_moneyflow_data(ts_code, trade_date, days_count)
    ##个股资金流向数据的参数列表
    moneyflow_param_list =['buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount', 'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
                           'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount', 'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
                           'net_mf_vol', 'net_mf_amount']
    ##存储个股资金流向数据的缩放倍数
    moneyflow_zoom_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0]
    if moneyflow_data.shape[0] < days_count:
        pass
    else:
        k = len(moneyflow_param_list)
        for i in range(k):
            for j in range(days_count):
                param = int(float(moneyflow_data[moneyflow_param_list[i]][j])*moneyflow_zoom_list[i])
                param_255 = get_int_to_4_255_gray(param)
                input_image[j*2+33][4*i]=int(param_255[0])
                input_image[j*2+33][4*i+1]=int(param_255[1])
                input_image[j*2+33][4*i+2]=int(param_255[2])
                input_image[j*2+33][4*i+3]=int(param_255[3])

    ##~~~~~~~~~备用行情
    days_count = 64
    bak_daily_data = get_n_bak_daily_data(ts_code, trade_date, days_count)
    ##个股备用数据的参数列表
    bak_daily_param_list =['pct_change', 'close', 'change', 'open', 'high', 'low', 'pre_close', 'vol_ratio',
                           'turn_over', 'swing', 'vol', 'amount', 'selling', 'buying', 'total_share', 'float_share',
                           'pe', 'float_mv', 'total_mv', 'avg_price', 'strength', 'activity', 'attack']
    ##存储个股备用数据的缩放倍数
    bak_daily_zoom_list = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                           100.0, 100.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0,
                           100.0, 100.0, 100.0, 100.0, 100.0, 1.0, 100.0]
    if bak_daily_data.shape[0] < days_count:
        pass
    else:
        k = len(bak_daily_param_list)
        for i in range(k):
            for j in range(days_count):
                param = int(float(bak_daily_data[bak_daily_param_list[i]][j])*bak_daily_zoom_list[i])
                param_255 = get_int_to_4_255_gray(param)
                input_image[j*2+33][4*i+128]=int(param_255[0])
                input_image[j*2+33][4*i+129]=int(param_255[1])
                input_image[j*2+33][4*i+130]=int(param_255[2])
                input_image[j*2+33][4*i+131]=int(param_255[3])

    ##~~~~~~~~~周线行情
    weeks_count = 56
    weekly_data = get_n_weekly_data(ts_code, trade_date, weeks_count)
    ##周线指标数据的参数列表
    weekly_data_param_list =['close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
    ##存储周线指标数据的缩放倍数
    weekly_data_zoom_list = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 10000.0, 1.0, 1.0/100, 1.0/10000]

    if weekly_data.shape[0] < weeks_count:
        pass
    else:
        k = len(weekly_data_param_list)
        for i in range(k):
            for j in range(weeks_count):
                param = int(float(weekly_data[weekly_data_param_list[i]][j])*weekly_data_zoom_list[i])
                param_255 = get_int_to_4_255_gray(param)
                input_image[j+160][4*i]=int(param_255[0])
                input_image[j+160][4*i+1]=int(param_255[1])
                input_image[j+160][4*i+2]=int(param_255[2])
                input_image[j+160][4*i+3]=int(param_255[3])

    ##~~~~~~~~~月线行情
    months_count = 40
    monthly_data = get_n_monthly_data(ts_code, trade_date, months_count)
    ##月线指标数据的参数列表
    monthly_data_param_list =['close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
    ##存储月线指标数据的缩放倍数
    monthly_data_zoom_list = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 10000.0, 1.0/100, 1.0/10000]
    if monthly_data.shape[0] < months_count:
        pass
    else:
        k = len(monthly_data_param_list)
        for i in range(k):
            for j in range(months_count):
                param = int(float(monthly_data[monthly_data_param_list[i]][j])*monthly_data_zoom_list[i])
                param_255 = get_int_to_4_255_gray(param)
                input_image[j+216][4*i]=int(param_255[0])
                input_image[j+216][4*i+1]=int(param_255[1])
                input_image[j+216][4*i+2]=int(param_255[2])
                input_image[j+216][4*i+3]=int(param_255[3])
    return input_image

def get_third_input_image(ts_code, trade_date):
    ##*******************第三通道其它数据
    input_image = np.zeros([256, 256, 1], dtype="uint8")
    if 'BJ' in ts_code:
        input_image[0][0] = 0
    if 'SH' in ts_code:
        input_image[0][0] = 128
    if 'SZ' in ts_code:
        input_image[0][0] = 255

    code_number = ts_code.split('.')
    code_number = int(code_number[0])
    code_number_255 = get_int_to_4_255_gray(code_number)
    input_image[0][4] = int(code_number_255[0])
    input_image[0][5] = int(code_number_255[1])
    input_image[0][6] = int(code_number_255[2])
    input_image[0][7] = int(code_number_255[3])

    trade_date = int(trade_date)
    trade_date_255 = get_int_to_4_255_gray(trade_date)
    input_image[0][8] = int(trade_date_255[0])
    input_image[0][9] = int(trade_date_255[1])
    input_image[0][10] = int(trade_date_255[2])
    input_image[0][11] = int(trade_date_255[3])
    
    return input_image

def get_input_image(ts_code, trade_date):
    image_a = get_first_input_image(ts_code, trade_date)
    image_b = get_second_input_image(ts_code, trade_date)
    image_c = get_third_input_image(ts_code, trade_date)
    image_input = cv2.merge([image_a, image_b, image_c])
    return image_input

def get_output_image(ts_code, trade_date):
    out_image = np.zeros([64, 64, 1], dtype="uint8")
    start_date = tools.get_delta_date(trade_date, -183)
    end_date = tools.get_delta_date(trade_date, 183)
    fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if fina_indicator_data.empty:
        pass
    else:
        fina_indicator_data = fina_indicator_data.fillna(-1)
        fina_indicator_data = fina_indicator_data.drop(fina_indicator_data[fina_indicator_data['ann_date']<trade_date].index)
        fina_indicator_data = fina_indicator_data.sort_values(by='end_date', ascending=True)
        nearest_fina_indicator_data = fina_indicator_data.iloc[0, :]
        # print(fina_indicator_data)
        for i in range(len(fina_indicator_param_list)):
            param = int(float(nearest_fina_indicator_data[fina_indicator_param_list[i]])*fina_indicator_zoom_list[i])
            param_255 = get_int_to_32_255_gray(param)
            m = int(i/2)
            n = i%2
            for j in range(32):
                out_image[m][n*32+j] = int(param_255[j])*255
    out_image = cv2.resize(out_image, [256, 256])
    return out_image

def resize_images(folder):
    image_names = tools.get_image_names(folder)
    for i in range(len(image_names)):
        print(float(i)/len(image_names))
        image_name = image_names[i]
        image = cv2.imread(image_name)
        image_resized = cv2.resize(image, (128, 128))
        cv2.imwrite(image_name, image_resized)

def rename_out_put_image(folder):
    image_names = tools.get_image_names(folder)
    for i in range(len(image_names)):
        print(float(i)/len(image_names))
        image_name = image_names[i]
        image_name_split = image_name.split('/')
        image_name_base = image_name_split[1]
        image_name_base_split = image_name_base.split('_')
        ts_code = image_name_base_split[0]
        trade_date = image_name_base_split[1]
        start_date = tools.get_delta_date(trade_date, -183)
        end_date = tools.get_delta_date(trade_date, 183)
        fina_indicator_data = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
        fina_indicator_data = fina_indicator_data.drop(
            fina_indicator_data[fina_indicator_data['ann_date'] < trade_date].index)
        fina_indicator_data = fina_indicator_data.sort_values(by='end_date', ascending=True)

        try:
            ann_date = fina_indicator_data['ann_date'][0]
            start_date = tools.get_delta_date(ann_date, -20)
            end_date = tools.get_delta_date(ann_date, 40)
            daily_data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if daily_data.empty:
                os.remove(image_name)
            else:
                try:
                    min_price = daily_data['close'].min()
                    # print(median_price)
                    k = int(min_price)
                    if k < 0:
                        k = 0.0
                    if k > 399:
                        k = 399
                    append_name = str(int(k))
                    image_name_new = image_name.replace('.png', '_' + append_name+'.png')
                    os.rename(image_name, image_name_new)
                except:
                    os.remove(image_name)
        except:
            os.remove(image_name)


def get_ts_code_list_today():
    ts_code_data_L = pro.stock_basic(exchange='', list_status='L')
    ts_code_data_D = pro.stock_basic(exchange='', list_status='D')
    ts_code_data = ts_code_data_L.append(ts_code_data_D)
    ts_code_data_P = pro.stock_basic(exchange='', list_status='P')
    ts_code_data = ts_code_data.append(ts_code_data_P)
    # 进行排序
    ts_code_data = ts_code_data.sort_values(by='list_date', ascending=True)
    ts_code_data = ts_code_data.reset_index(drop=True)
    return ts_code_data['ts_code']

def get_ts_code_list(trade_date):
    ts_code_data_L = pro.stock_basic(exchange='', list_status='L')
    ts_code_data_D = pro.stock_basic(exchange='', list_status='D')
    ts_code_data = ts_code_data_L.append(ts_code_data_D)
    ts_code_data_P = pro.stock_basic(exchange='', list_status='P')
    ts_code_data = ts_code_data.append(ts_code_data_P)
    # 进行排序
    ts_code_data = ts_code_data.sort_values(by='list_date', ascending=True)
    ts_code_data = ts_code_data.reset_index(drop=True)
    ts_code_data = ts_code_data.drop(ts_code_data[ts_code_data['list_date']>trade_date].index)
    return ts_code_data['ts_code']


def get_train_images():
    input_folder = 'input'
    output_folder = 'output'
    print(os.getcwd())
    if (os.path.exists(input_folder)) == False:
        os.mkdir(input_folder)
    if (os.path.exists(output_folder)) == False:
        os.mkdir(output_folder)
    input_folder = os.getcwd() + '//' + 'input'
    output_folder = os.getcwd() + '//' + 'output'
    #随机获取1995-2005年训练数据
    trade_date_list = get_trade_date_list_period('19950101', '20221231')
    count = 0
    while True:
        print(count)
        try:
            i = random.randint(0, len(trade_date_list))
            trade_date = trade_date_list[i]
            print(trade_date)
            ts_code_list = get_ts_code_list(trade_date)
            j = random.randint(0, len(ts_code_list))
            ts_code = ts_code_list[j]
            print(ts_code)
            input_image_name = input_folder + '//' + ts_code + '_' + trade_date + '_in.png'
            output_image_name = output_folder + '//' + ts_code + '_' + trade_date + '_out.png'
            if os.path.exists(input_image_name):
                pass
            else:
                input_image = get_input_image(ts_code, trade_date)
                cv2.imwrite(input_image_name, input_image)
                output_image = get_output_image(ts_code, trade_date)
                cv2.imwrite(output_image_name, output_image)
        except:
            pass
        count = count + 1
get_train_images()
# rename_out_put_image('valuation3')
# resize_images('valuation3')