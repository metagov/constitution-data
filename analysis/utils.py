import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
from airtable import airtable
from sklearn.preprocessing import MultiLabelBinarizer

# Figure formatting settings
DEFAULT_COLOR = '#66C2A5'
DEFAULT_SIZE = (5, 5)
sns.set(rc={"figure.figsize":DEFAULT_SIZE})
sns.set(font_scale=1.25)

if not os.path.isdir("figures"):
    os.mkdir("figures")


def load_airtable():
    """Load Airtable base. Requires editor access to the Airtable and saving the corresponding API key to api_key.txt"""
    
    BASE_ID = 'appx3e9Przn9iprkU' # The Govbase base ID
    with open(os.path.join('..', 'api_key.txt'), 'r') as f:
        API_KEY = f.readline().strip()
        
    return airtable.Airtable(BASE_ID, API_KEY)


def load_table_as_df(at, tableName, kwargs=None):
    """Get all records in a table and load into DataFrame"""
    
    if kwargs is None:
        kwargs = {}
    
    # Get all records
    records = []
    for r in at.iterate(tableName, **kwargs):
        records.append({'id': r['id'], **(r['fields'])})
        
    # Convert to DataFrame
    df = pd.DataFrame(records)
    df.set_index('id', inplace=True)
    
    return df


def plot_coded_column(df_all, col, label='', orient='h', size=None, plotType='bar'):
    """Plot frequency of unique list items for coded columns
    Handle columns with list values differently from columns with single values"""
    
    df = df_all.copy(deep=True)
    isListCol = any([isinstance(d, list) for d in df[col]])
    
    if isListCol:
        # Make sure all values in column are lists
        df[col] = df[col].apply(lambda d: d if isinstance(d, list) else [])

        # One-hot encode column of lists
        mlb = MultiLabelBinarizer(sparse_output=True)
        df_onehot = pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(df[col]),
            index=df.index,
            columns=mlb.classes_)

        # Get count for each unique item
        df_sum = pd.DataFrame(df_onehot.sum()).sort_values(0, axis=0, ascending=False).transpose()
    else:
        df_sum = pd.DataFrame(df[col].value_counts()).transpose()
    
    # Resize plot if needed
    if size is not None:
        sns.set(rc={"figure.figsize": size})
        sns.set(font_scale=1.25)
    
    plt.figure()
    
    if plotType == 'bar':
        # Plot bar chart of unique list items
        ax = sns.barplot(data=df_sum, orient=orient, color=DEFAULT_COLOR)

        # Formatting
        if orient == 'h':
            plt.ylabel(label)
            labels = ["\n".join(textwrap.wrap(c, width=30)) for c in df_sum.columns]
            ax.set(yticklabels=labels)
            plt.xlabel('Count')
            if not isListCol:
                plt.xlim((0, df_sum.sum().sum() + 1))
        else:
            plt.ylabel('Count')
            plt.xlabel("\n".join(textwrap.wrap(label, 50)))
            labels = ["\n".join(textwrap.wrap(c, width=25)) for c in df_sum.columns]
            ax.set(xticklabels=labels)
            if not isListCol:
                plt.ylim((0, df_sum.sum().sum() + 1))
    
    elif plotType == 'pie':
        # Plot pie chart of unique list items
        fig = plt.pie(df_sum.squeeze(), labels=df_sum.columns, 
                      colors=sns.color_palette("Set2"), wedgeprops=dict(width=0.5))
        plt.title(label)
    
    if size is not None:
        sns.set(rc={"figure.figsize": DEFAULT_SIZE})
        sns.set(font_scale=1.25)

    plt.savefig(os.path.join("figures", f"{col}.jpeg"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join("figures", f"{col}.pdf"), bbox_inches='tight')
