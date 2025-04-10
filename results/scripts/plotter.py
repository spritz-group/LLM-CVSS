import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
legend_fontsize = 16


def generate_embeddings_csv():
    embedders = {
        "ATTACK-BERT": "ATTACK-BERT",
        "nomic": "nomic",
        "all-MiniLM-L6-v2": "all-MiniLM",
    }

    dataframes = {}
    for key, modelname in embedders.items():
        file_path = f"../embeddings/base/{key}.csv"
        df = pd.read_csv(file_path)
        df = df[df['model'] == 'xgboost']
        df['source'] = modelname  
        dataframes[key] = df

    merged_dataframe = pd.concat(dataframes.values(), ignore_index=True)

    merged_dataframe['source'] = pd.Categorical(merged_dataframe['source'], categories=embedders.values())
    merged_dataframe = merged_dataframe.drop(columns=['model']).rename(columns={'source': 'Embedding'})
    columns = ['Embedding'] + [col for col in merged_dataframe.columns if col != 'Embedding']
    merged_dataframe = merged_dataframe[columns]

    merged_dataframe.to_csv("embeddings_xgboost.csv", index=False)


#######################################################

def llm_prompt_eng():
    dataframes = {}

    files = {
        "../vector.csv": "Base", 
        "../vector_few_shot.csv": "Few Shot", 
        "../cwe_vector.csv": "CWE",
        "../components.csv": "Components",
        }

    for key, element in files.items():
        df = pd.read_csv(key)
        df = df[df['model'] == 'gemma3_12b']
        df['source'] = element
        dataframes[key] = df


    merged_dataframe = pd.concat(dataframes.values(), ignore_index=True)
    merged_dataframe = merged_dataframe.drop(columns=['model']).rename(columns={'source': 'Improvement'})
    columns = ['Improvement'] + [col for col in merged_dataframe.columns if col != 'Improvement']
    merged_dataframe = merged_dataframe[columns]
    merged_dataframe.to_csv("vector_gemma_improvements.csv", index=False)

    plot_data = merged_dataframe.set_index('Improvement').T
    plot_data.plot(kind='bar', figsize=(10, 6), width=0.7, zorder=5)

    plt.xlabel('Elements')
    plt.ylabel('Accuracy')
    #plt.title('Improvement Comparison Across Metrics')
    plt.legend(title='Engineering', loc='best', ncol=2, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
    plt.xticks(rotation=0)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig("prompt-eng.pdf")
    plt.show()

########################################################

def comparison():

    llm_df = pd.read_csv("../vector.csv")
    llm_df = llm_df[llm_df['model'] == 'gemma3_12b']
    llm_df['model'] = llm_df['model'].replace('gemma3_12b', 'Vanilla')

    attack_bert_df = pd.read_csv("../embeddings/base/all-MiniLM-L6-v2.csv")
    attack_bert_df = attack_bert_df[attack_bert_df['model'] == 'xgboost']
    attack_bert_df['model'] = attack_bert_df['model'].replace('xgboost', 'Embeddings')

    # Merge the two datasets
    merged_df = pd.merge(llm_df, attack_bert_df, on=llm_df.columns.tolist(), how='outer', suffixes=('_LLM', '_BERT'))
    merged_df['model'] = pd.Categorical(merged_df['model'], categories=['Vanilla', 'Embeddings'], ordered=True)
    merged_df = merged_df.sort_values('model')

    # Prepare data
    plot_data = merged_df.set_index('model').T  
    # Plot 
    plot_data.plot(kind='bar', figsize=(12, 7), width=0.8, zorder=5)

    plt.xlabel('Elements')
    plt.ylabel('Accuracy')
    plt.legend(title='Method', loc='best', fontsize=legend_fontsize, title_fontsize=legend_fontsize)
    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("comparison_plot.pdf")
    plt.show()

    # Calculate and print the average of each row
    merged_df = merged_df.drop(columns=['model'])  
    row_averages = merged_df.mean(axis=1)
    print(f"Row averages: {row_averages}")

    llm_subset = list(llm_df[['AV', 'AC', 'UI']].iloc[0])
    attack_bert_subset = list(attack_bert_df[["PR", "S", "C", "I", "A"]].iloc[0])
    combined_list = llm_subset + attack_bert_subset
    average = sum(combined_list) / len(combined_list)
    print("Best average:", average)
    
    

#########################################################

def embeddings_enhancements():
    dataframes = {}

    files = {
        "../embeddings/base/all-MiniLM-L6-v2.csv": "allMiniLM-base", 
        "../embeddings/enhanced/all-MiniLM-L6-v2.csv": "allMiniLM-CWE", 
        "../embeddings/base/ATTACK-BERT.csv": "ATTACK-BERT-base", 
        "../embeddings/enhanced/ATTACK-BERT.csv": "ATTACK-BERT-CWE", 
        }

    for key, element in files.items():
        df = pd.read_csv(key)
        df = df[df['model'] == 'xgboost']
        df['source'] = element
        dataframes[key] = df

    merged_dataframe = pd.concat(dataframes.values(), ignore_index=True)
    merged_dataframe = merged_dataframe.drop(columns=['model']).rename(columns={'source': 'Improvement'})
    columns = ['Improvement'] + [col for col in merged_dataframe.columns if col != 'Improvement']
    merged_dataframe = merged_dataframe[columns]
    merged_dataframe.to_csv("vector_xgboost_improvements.csv", index=False)

    plot_data = merged_dataframe.set_index('Improvement').T
    plot_data.plot(kind='bar', figsize=(10, 6), width=0.7, zorder=5)

    # Add labels and title
    plt.xlabel('Elements')
    plt.ylabel('Accuracy')
    #plt.title('Improvement Comparison Across Metrics')
    plt.legend(title='Model-enhancement', loc='lower center', ncol=2, framealpha=1, fontsize=legend_fontsize, title_fontsize=legend_fontsize)
    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("emb-enhancements.pdf")
    plt.show()

#############################################

def confusion_matrix():
    filename, col_names = "../embeddings/base/confusion_matrices/ATTACK-BERT_xgboost_AV.txt", ["A", "L", "N", "P"]

    df = pd.read_csv(filename, sep=" ", header=None)

    df.columns = col_names
    df.index = col_names

    plt.figure(figsize=(10, 10))
    sns.heatmap(df, annot=True, fmt="d", cmap="Reds", cbar=False, xticklabels=True, yticklabels=True, square=True)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    #plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("confusion_matrix.pdf")
    plt.show()

################################################
if __name__ == "__main__":
    #llm_prompt_eng()
    #embeddings_enhancements()
    comparison()
    #confusion_matrix()
    #generate_embeddings_csv()
