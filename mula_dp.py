import copy, random
import pickle as pkl

from typing import List, Tuple
from src.tools.utils import *
from global_values import *
from src.model.mula_tabpro import *
from src.tools.logger import Logger
from src.data import TQAData
from src.model.mula_tabpro.agent import ViewGenerator, Imputater, ColTypeDeducer, Cleaner, Coder, Planner, Normalizer, Augmenter, Filter

class MultiAgentDataPrep:
    def __init__(self, llm_name:str, logger_root='tmp/table_llm_log', logger_file=f'mula_tabpro_v{TABLELLM_VERSION}.log', temp_data_path = MULTIA_TEMP_DATA_PATH):
        self.llm_name = llm_name
        self.view_gen = ViewGenerator(llm_name=LLM_NAME, logger_root=logger_root, logger_file=logger_file)
        self.imputater = Imputater(llm_name=LLM_NAME, logger_root=logger_root, logger_file=logger_file)
        self.coltype_deducer = ColTypeDeducer(llm_name=LLM_NAME, logger_root=logger_root, logger_file=logger_file)
        self.cleaner = Cleaner(llm_name=LLM_NAME, logger_root=logger_root, logger_file=logger_file)
        self.logger = Logger(name='MultiAgentDataPrep', root=logger_root, log_file=logger_file)
        self.coder = Coder(llm_name=LLM_NAME, logger_root=logger_root, logger_file=logger_file)
        self.planner = Planner(llm_name=LLM_NAME, logger_root=logger_root, logger_file=logger_file)
        self.normalizer = Normalizer(llm_name=LLM_NAME, logger_root=logger_root, logger_file=logger_file)
        self.augmenter = Augmenter(llm_name=LLM_NAME, logger_root=logger_root, logger_file=logger_file)
        self.filter = Filter(llm_name=LLM_NAME, logger_root=logger_root, logger_file=logger_file)
        self.op2coder = Op2Coder(llm_name=LLM_NAME, logger_root=logger_root, logger_file=logger_file)

        self.temp_data_path = temp_data_path
        self.self_corr_inses = []
        self.icl_inses = []

        self.temp_relcol_mapreq_analysissketchsql = {}
        if os.path.exists(self.temp_data_path):
            try:
                self.temp_relcol_mapreq_analysissketchsql = pkl.load(open(self.temp_data_path, 'rb'))
                self.logger.log_message(msg=f'I: Load temp data from {self.temp_data_path}, {len(self.temp_relcol_mapreq_analysissketchsql)} records loaded.')
            except Exception as e:
                self.temp_relcol_mapreq_analysissketchsql = {}
                self.logger.log_message(level='error', msg=f'E: Error raised in loading temp data: {e}')

    def _generate_related_cols_and_mapping_requirements(self, data:TQAData, instance_pool=None, GEN_COL_FLAG=True, analysissketch_vote_cnt=QASKETCH_VOTE, load_from_temp=True):
        #? Generate multiple analysissketch_sql, then select the union for related_column, and update the mapping requirements
        #? Update rules: select the mapping requirements with shorter length.
        #* add function that allow to load temp data from disk
        if load_from_temp and data.id in self.temp_relcol_mapreq_analysissketchsql:
            tar_dic = self.temp_relcol_mapreq_analysissketchsql[data.id]
            self.logger.log_message(msg=f'/*************** Return temp data for {data.id} ***************/')
            return tar_dic['related_columns'], tar_dic['mapping_requirements'], tar_dic['analysissketch_sqls']

        related_columns = []
        mapping_requirements = None
        sql_with_mp = None
        analysissketch_sqls = []
        for _ in range(analysissketch_vote_cnt):
            post_sql, rel_cols, map_requires = self.analysissketch.process(data, instance_pool)
            if len(rel_cols) == 0:
                rel_cols = list(data.tbl.columns)

            analysissketch_sqls.append(post_sql)
            related_columns = list(set(related_columns + rel_cols))

            if mapping_requirements is None:
                mapping_requirements = map_requires
                sql_with_mp = post_sql
            if len(map_requires) < len(mapping_requirements):
                mapping_requirements = map_requires
                sql_with_mp = post_sql
                
        if GEN_COL_FLAG:
            related_columns = self._update_related_cols(related_columns, mapping_requirements, sql_with_mp)

        return related_columns, mapping_requirements, analysissketch_sqls

    def _get_coltype_dict(self, data:TQAData, related_columns:List[str], analysissketch_sql:str, coltype_vote=3, instance_pool=None):
        coltype_dict = {}
        for _ in range(coltype_vote):
            try:
                colty_dic = self.coltype_deducer.process(data, related_columns, analysissketch_sql, instance_pool)
            except:
                continue
            for col, ctype in colty_dic.items():
                if col not in coltype_dict:
                    coltype_dict[col] = []
                coltype_dict[col].append(ctype)
        for col in coltype_dict:
            coltype_dict[col] = max(coltype_dict[col], key=coltype_dict[col].count)
        return coltype_dict

    def process(self, data:TQAData):
        self.data = data
        original_data = copy.deepcopy(data)
        self.log_info = {}
        self.self_corr_inses = []
        original_cols = list(self.data.tbl.columns)
        self.log_info['original_table_schema'] = original_cols
        self.log_info['title'] = data.title
        self.log_info['question'] = data.question

        # 0. base_cleaning
        self.data.tbl,_ = base_clean_dataframe(self.data.tbl, colname_cleaning=True, value_cleaning=not (TASK_TYPE=='birdqa'), 
                                               column_differ_keyword=True, value_standardization=True, numeralize=True, dataformat=False)

        start_time = time.time()
        # 1. generate logical ops
        post_sql, rel_cols, chains = self.planner.process(self.data)
        self.log_info['qa_sketch'] = post_sql
        self.log_info['related_columns'] = rel_cols
        self.log_info['logical_ops'] = [str(op) for op in chains]

        # 2. implement logical ops: generate physical ops and execute them
        physical_chains = []
        for op in chains:
            physical_op = None
            if op.type == 'Augment':
                try:
                    in_data = copy.deepcopy(self.data)
                    self.data, physical_op = self.augmenter.implement_augment(self.data, op)
                except Exception as e:
                    if CODE_ASSIT_OP:
                        self.logger.log_message(level='error', msg=f'E: Error raised in implementing Augment: {e}')
                        try:
                            self.data, physical_op = self.op2coder.implement_logical_operator(copy.deepcopy(in_data), op)
                            physical_op.args['in_columns'] = df_to_cotable(in_data.tbl, cut_line=DEFAULT_ROW_CUT) if in_data.tbl is not None else ""
                            physical_op.args['out_columns'] = df_to_cotable(self.data.tbl, cut_line=DEFAULT_ROW_CUT) if self.data.tbl is not None else ""
                        except Exception as e:
                            pass

                if physical_op is not None:
                    physical_chains.append(physical_op)

            elif op.type == 'Normalize':
                try:
                    in_data = copy.deepcopy(self.data)
                    self.data, physical_op = self.normalizer.implement_normalize(self.data, op)
                except Exception as e:
                    if CODE_ASSIT_OP:
                        self.logger.log_message(level='error', msg=f'E: Error raised in implementing Normalize: {e}')
                        try:
                            self.data, physical_op = self.op2coder.implement_logical_operator(copy.deepcopy(in_data), op)
                            physical_op.args['in_columns'] = df_to_cotable(in_data.tbl, cut_line=DEFAULT_ROW_CUT) if in_data.tbl is not None else ""
                            physical_op.args['out_columns'] = df_to_cotable(self.data.tbl, cut_line=DEFAULT_ROW_CUT) if self.data.tbl is not None else ""
                        except Exception as e:
                            pass
                
                if physical_op is not None:
                    physical_chains.append(physical_op)

        # 3. generate filter op
        filter_cols = copy.deepcopy(rel_cols)
        # 3.1 remove all columns that only exist in src_cols of augment op
        tmp_sql = copy.deepcopy(post_sql)
        qa_pattern = 'MAP\(\[.*?\],.*?\)'
        qas = re.findall(qa_pattern, tmp_sql)
        for idx, qa in enumerate(qas):
            tmp_sql = tmp_sql.replace(qa, f"placeholder{idx}")

        # 3.2 add all new generated columns
        for op in physical_chains:
            if 'new_column' in op.args:
                filter_cols.append(op.args['new_column'])
            if 'new_column_lis' in op.args:
                for col in op.args['new_column_lis']:
                    filter_cols.append(col)
            if 'source_cols' in op.args:
                src_cols = op.args['source_cols']
                for col in src_cols:
                    if col in filter_cols and col not in tmp_sql:
                        filter_cols.remove(col)

        filterop = FilterOp(req='Nan', cols=filter_cols)
        self.data, op = self.filter.implement_filter(self.data, filterop)
        chains.append(filterop)
        physical_chains.append(op)

        # update self.log_info
        self.log_info['logical_ops'] = [str(op) for op in chains]
        self.log_info['physical_ops'] = [op.op_string() for op in physical_chains]
        self.log_info['related_columns'] = filter_cols

        self.self_corr_inses = []
        self.icl_inses = []

        self.logger.log_message(msg='\n'.join(['/*************** After Data Prep for data ***************/',
                                               f'【title】{data.title}',
                                               f'【Question】{data.question}',
                                               f'【QA Sketch】{self.log_info["qa_sketch"]}',
                                               f'【Related Columns:】{self.log_info["related_columns"]}',
                                               f'【Logical Operators:】{self.log_info["logical_ops"]}']))
        
        self.logger.log_message(msg='\n'.join([f'******************** id: {data.id}, question: {data.question} ********************',
                                               f'【Original Table】\n{original_data.tbl}',
                                               f'【Processed Table】\n{self.data.tbl}']))

        return self.data, self.log_info

    def _update_related_cols(self, related_cols: List[str], mapping_requirements: List[Tuple[str, str]], sql:str):
        col_exists_cnt_map = {}

        for _, cols in mapping_requirements:
            for col in cols:
                if col not in col_exists_cnt_map:
                    col_exists_cnt_map[col] = 0
                col_exists_cnt_map[col] += 1

        for col in col_exists_cnt_map:
            # if the column only exists in the mapping requirements, then remove it from the related columns
            if col_exists_cnt_map[col] == sql.count(f'`{col}`'):
                if col in related_cols:
                    related_cols.remove(col)

        return related_cols
