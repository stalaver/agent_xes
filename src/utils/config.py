# config.py - Centralized experiment configuration

CONFIG = {
    # Data Collection
    'webarena': {
        'base_url': 'http://localhost',
        'sites': ['shopping', 'shopping_admin', 'reddit', 'gitlab', 
                  'wikipedia', 'map', 'reddit2', 'shopping_map'],
        'holdout_sites': ['gitlab', 'wikipedia'],  # Cross-site validation
    },
    
    # Models
    'models': {
        'llama': 'meta-llama/Llama-3.2-3B-Instruct',
        'qwen': 'Qwen/Qwen2.5-7B-Instruct',
        'mistral': 'mistralai/Mistral-7B-Instruct-v0.3',
    },
    
    # Dataset targets
    'dataset': {
        'webarena_total': 1500,
        'browsergym_total': 500,
        'balanced_subset': 1000,  # 500 success + 500 failure
        'splits': {'train': 0.6, 'val': 0.2, 'test': 0.2},
    },
    
    # Pattern Mining
    'mining': {
        'k_values': [5, 8, 10, 15],
        'min_support': {
            'per_site': 0.03,
            'global': 0.01,
        },
        'algorithm': 'BIDE',
        'max_pattern_length': 5,
        'minimum_sites': 2,  # Coverage requirement
    },
    
    # Symbolization
    'symbolization': {
        'levels': [0, 1, 2],  # Fine, Medium, Coarse
        'primary_level': 1,   # Medium as default
    },
    
    # Evaluation
    'evaluation': {
        'primary_k': 10,
        'success_criteria': {
            'f1_at_k': 0.65,
            'auc_pr': 0.75,
            'cross_site_delta': 0.10,
            'cohens_kappa': 0.70,
        },
    },
    
    # Baselines
    'baselines': [
        'frequency_vector',
        'ngram',
        'taspm',
        'process_conformance',
        'deeplog',
        'bilstm_smote',
    ],
    
    # Paths
    'paths': {
        'raw_traces': 'data/raw_traces',
        'processed': 'data/processed',
        'patterns': 'data/patterns',
        'results': 'experiments/results',
        'spmf_jar': '/opt/spmf/spmf.jar',
    },
}