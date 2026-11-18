import sys, random
import threading
import pickle as pkl
from joblib import Parallel, delayed

from mula_dp import MultiAgentDataPrep
from src.tools.binder_utils.evaluator import Evaluator
from src.dataset import TQADataset, TFVDataset
from src.model.mula_tabpro import *
from src.model.mula_tabpro.base import *
from src.tools.utils import *
from global_values import *

class Experiment:
    def __init__(self, mulprocess_cnt, llm_name, VERSION=TABLELLM_VERSION):
        self.mulprocess_cnt = mulprocess_cnt
        self.llm_name = llm_name
        self.VERSION = VERSION

        if not os.path.exists(f'{TMP_DIR}/outputs'):
            os.makedirs(f'{TMP_DIR}/outputs')
        if not os.path.exists(f'{TMP_DIR}/outputs/analyze'):
            os.makedirs(f'{TMP_DIR}/outputs/analyze')
        if not os.path.exists(f'{TMP_DIR}/table_llm_log'):
            os.makedirs(f'{TMP_DIR}/table_llm_log')

    def run_multi_agents_data_prep(self, sample_num):
        BASE_VERSION=TABLELLM_VERSION+f'_{sample_num}' + '_data_prep'
        print(f'BASE_VERSION: {BASE_VERSION}')
        datas = self.load_datas(sample_num)
        print(f'len(datas): {len(datas)}')
        processed_datas = self.multi_agents_data_prep(datas, save=True, BASE_VERSION=BASE_VERSION)
        self.nl2sql_baseline(processed_datas, save=True, BASE_VERSION=BASE_VERSION + '_run_nl2sql', mode='prep')
        self.evaluate_result(BASE_VERSION + '_run_nl2sql', analyze=True)

    def run_nl2sql_baseline(self, sample_num):
        BASE_VERSION=TABLELLM_VERSION+f'_{sample_num}' + '_baseline_nl2sqler'
        print(f'BASE_VERSION: {BASE_VERSION}')
        datas = self.load_datas(sample_num)
        print(f'len(datas): {len(datas)}')
        self.nl2sql_baseline(datas, save=True, BASE_VERSION=BASE_VERSION)
        self.evaluate_result(BASE_VERSION, analyze=True)

    def evaluate_result(self, VERSION, analyze=True, data_dic=None, allow_semantic=True):
        binder_eval = Evaluator()
        result_path = f'{TMP_DIR}/outputs/result_v{VERSION}.json'
        reses = open_json(result_path)
        err_ids = []
        eval_results = {'correct': 0, 'total': 0, 'accuracy': 0.0}
        hit_dic = {}
        analyze_text = ''
        for id in tqdm(reses):
            eval_results['total'] += 1
            rec = reses[id]

            label = rec['label']
            pred = rec['answer']

            if type(pred) == list:
                pred = '|'.join([str(p) for p in pred])
            
            if binder_eval.evaluate(
                pred.split('|') if type(pred) == str else pred,
                label.split('|') if type(label) == str else label,
                'tab_fact' if TASK_TYPE == 'tablefact' else 'wikitq',
                allow_semantic=allow_semantic,
                question=rec['question']):

                eval_results['correct'] += 1
                eval_results
            else:
                err_ids.append(id)
                hit_dic[id] = False

                if analyze:
                    sql = rec['sql'] if 'sql' in rec else 'N/A'
                    table = rec['table'] if data_dic is None else df_to_cotable(data_dic[id].tbl)
                    analyze_text += '\n'.join([f"【Table】: {id}\n{table}", f"【Question】: {rec['question']}", f"【Label】: {label}", f"【Prediction】: {pred}", f"【SQL】\n{sql}", f"【Reason】:", '', ''])
                pass

        if analyze:
            if not os.path.exists(f'{TMP_DIR}/outputs/analyze'):
                os.makedirs(f'{TMP_DIR}/outputs/analyze')
            with open(f'{TMP_DIR}/outputs/analyze/analyze_v{VERSION}.txt', 'w', encoding='utf-8') as f:
                f.write(str(err_ids) + '\n\n' + analyze_text)

        if eval_results['total'] > 0:
            eval_results['accuracy'] = eval_results['correct'] / eval_results['total']
        print(VERSION, eval_results, sep='\t\t\t')
        save_json(eval_results, f'{TMP_DIR}/outputs/eval_results_v{VERSION}.json')
        return hit_dic
    
    def load_datas(self, sample_num=-1, split=SPLIT):

        if TASK_TYPE == 'tableqa':
            if not os.path.exists(rf'{TMP_DIR}/{TASK_TYPE}_{split}_{sample_num}.pkl'):
                ds = pkl.load(open(rf'{TMP_DIR}/wiki_qa.pkl', 'rb'))
                datas = ds.train_data if split == 'train' else ds.test_unseen_data
                if sample_num != -1 and sample_num < len(datas):
                    datas = random.sample(datas, sample_num)
                # save as pkl
                pkl.dump(datas, open(rf'{TMP_DIR}/{TASK_TYPE}_{split}_{sample_num}.pkl', 'wb'))
            else:
                # read pkl
                print(f'load data from pkl')
                datas = pkl.load(open(rf'{TMP_DIR}/{TASK_TYPE}_{split}_{sample_num}.pkl', 'rb'))

        elif TASK_TYPE == 'tablefact':
            if not os.path.exists(rf'{TMP_DIR}/{TASK_TYPE}_{split}_{sample_num}.pkl'):
                ds = TFVDataset(dataset_name='tabfact')
                # ds = pkl.load(open(rf'{TMP_DIR}/tab_fact.pkl', 'rb'))
                ds.load_data()
                datas = ds.train_data if split == 'train' else ds.test_data
                if sample_num != -1 and sample_num < len(datas):
                    datas = random.sample(datas, sample_num)
                # save as pkl
                pkl.dump(datas, open(rf'{TMP_DIR}/{TASK_TYPE}_{split}_{sample_num}.pkl', 'wb'))
            else:
                # read pkl
                print(f'load data from pkl')
                datas = pkl.load(open(rf'{TMP_DIR}/{TASK_TYPE}_{split}_{sample_num}.pkl', 'rb'))
        else:
            raise Exception(f'Unknown task type: {TASK_TYPE}')
        
        print(f'Total data: {len(datas)}')

        return datas

    def _nl2sql_one_data(self, data, llm_name, VERSION, mode='base'):
        nl2sqler = NL2SQLer(llm_name=llm_name, logger_file=f'mula_tabpro_v{VERSION}.log', mode=mode)

        try:
            sql, pro_tbl = nl2sqler.process(data=data)
            answer = extract_answers(pro_tbl)
        except Exception as e:
            sql = nl2sqler.sql
            answer = nl2sqler.ans
        
        self_corr_inses = nl2sqler.self_corr_inses
        icl_inses = nl2sqler.icl_inses

        return data.id, sql, answer, self_corr_inses, icl_inses

    def nl2sql_baseline(self, datas, save=True, BASE_VERSION=TABLELLM_VERSION, mode='base'):
        json_result = {}
        
        if os.path.exists(f'{TMP_DIR}/outputs/result_v{BASE_VERSION}.json'):
            json_result = open_json(f'{TMP_DIR}/outputs/result_v{BASE_VERSION}.json')
            print(len(json_result))
        
        logger = Logger(name='EXPERIMENT', log_file=f'mula_tabpro_v{BASE_VERSION}.log', root='tmp/table_llm_log')

        data_dic = {data.id: data for data in datas}

        for i in tqdm(range(0, len(datas), SAVE_STEP)):

            results = Parallel(n_jobs=self.mulprocess_cnt, require='sharedmem')(
                delayed(self._nl2sql_one_data)(data, self.llm_name, BASE_VERSION, mode) for data in datas[i:i+SAVE_STEP]\
                    if data.id not in json_result
            )

            for res in results:
                id, sql, answer, _, _ = res
                if 'answer' == 'INIT':
                    answer = 0
                data = data_dic[id]
                
                json_result[id] = {
                    'table': df_to_cotable(data.tbl, cut_line=DEFAULT_ROW_CUT),
                    'question': data.question,
                    'label': data.label,
                    'sql': sql,
                    'answer': answer
                }

                logger.log_message(msg=f'******************** #num: {len(json_result)}, id: {id}, label: {data.label}, answer: {answer} ********************')

                if save:
                    save_json(json_result, f'{TMP_DIR}/outputs/result_v{BASE_VERSION}.json')

        return json_result

    def _prepare_one_data(self, data, llm_name, VERSION):
        """The code for preparing one data with multi-agents.

        Args:
            data (TQAData): the data to be processed.
            llm_name (str): the name of the language model.
            VERSION (str): the version of the experiment.
            GEN_COL_FLAG (bool, optional): whether use Generator or not. Defaults to True.
            CLEAN_FLAG (bool, optional): whether use Cleaner or not. Defaults to True.
            IMPUTATE_FLAG (bool, optional): whether use Imputator or not. Defaults to True.

        Returns:
            processed_data (TQAData): the processed data.
            log_info (dict): the log information.
        """
        process_id = threading.get_ident()
        original_data = copy.deepcopy(data)
        mdp = MultiAgentDataPrep(llm_name=llm_name, logger_file=f'process{process_id}.log', logger_root=f'tmp/table_llm_log/mula_tabpro_v{VERSION}')
        try:
            processed_data, log_info = mdp.process(data)
        except Exception as e:
            mdp.logger.log_message(msg = f'!!!!!!!!!!!!!!!!!!!Error for {data.id}!!!!!!!!!!!!!!!!!: {e}')
            processed_data = mdp.data
            log_info = mdp.log_info
            if 'related_columns' in log_info:
                rel_cols = log_info['related_columns']
                processed_data.tbl = original_data.tbl[[x for x in rel_cols if x in original_data.tbl.columns]]

        self_corr_inses = mdp.self_corr_inses
        icl_inses = mdp.icl_inses

        return processed_data, log_info, self_corr_inses, icl_inses

    def multi_agents_data_prep(self, datas, save=True, BASE_VERSION=TABLELLM_VERSION):
        processed_datas = []
        log_infos = {}
        mul_dp_temp_data = {}

        logger = Logger(name='EXPERIMENT', log_file=f'mula_tabpro_v{BASE_VERSION}.log', root=f'tmp/table_llm_log/mula_tabpro_v{BASE_VERSION}')
        if os.path.exists(rf'{TMP_DIR}/outputs/mula_tabpro_v{BASE_VERSION}_processed_data.pkl'):
            with open(rf'{TMP_DIR}/outputs/mula_tabpro_v{BASE_VERSION}_processed_data.pkl', 'rb') as f:
                processed_datas = pkl.load(f)
            logger.log_message(msg=f'processed_datas: {len(processed_datas)}')
            
        if os.path.exists(rf'{TMP_DIR}/outputs/mula_tabpro_v{BASE_VERSION}_log_infos.json'):
            log_infos = open_json(rf'{TMP_DIR}/outputs/mula_tabpro_v{BASE_VERSION}_log_infos.json')
            logger.log_message(msg=f'log_infos: {len(log_infos)}')

        if os.path.exists(MULTIA_TEMP_DATA_PATH):
            with open(MULTIA_TEMP_DATA_PATH, 'rb') as f:
                mul_dp_temp_data = pkl.load(f)

        # save results every $SAVE_STEP datas
        for i in tqdm(range(0, len(datas), SAVE_STEP)):
            results = Parallel(n_jobs=self.mulprocess_cnt, require='sharedmem')(
                delayed(self._prepare_one_data)(data, self.llm_name, BASE_VERSION) for data in datas[i:i+SAVE_STEP]\
                    if data.id not in log_infos and data.id not in [d.id for d in processed_datas]
            )

            for res in results:
                if res is None:
                    continue
                processed_data, log_info, self_corr_inses, icl_inses = res
                processed_datas.append(processed_data)
                log_infos[processed_data.id] = log_info

                logger.log_message(msg=f'******************** #num: {len(processed_datas)}, id: {processed_data.id}, question: {processed_data.question}, len(self_corr_inses): {len(self_corr_inses)} ********************')

                if save:
                    if 'relcol_mapreq_analysissketchsql' in log_info:
                        analysissketch_result_dic = log_info['relcol_mapreq_analysissketchsql']
                        log_info.pop('relcol_mapreq_analysissketchsql')
                        if processed_data.id not in mul_dp_temp_data:
                            mul_dp_temp_data[processed_data.id] = analysissketch_result_dic
                            with open(MULTIA_TEMP_DATA_PATH, 'wb') as f:
                                pkl.dump(mul_dp_temp_data, f)
                    # Save as pkl
                    with open(rf'{TMP_DIR}/outputs/mula_tabpro_v{BASE_VERSION}_processed_data.pkl', 'wb') as f:
                        pkl.dump(processed_datas, f)
                    # Save as json
                    save_json(log_infos, rf'{TMP_DIR}/outputs/mula_tabpro_v{BASE_VERSION}_log_infos.json')

        return processed_datas

if __name__ == '__main__':
    exp = Experiment(mulprocess_cnt=MULPROCESS_CNT, llm_name=LLM_NAME)
    exp.run_multi_agents_data_prep(sample_num=SAMPLE_NUM)
