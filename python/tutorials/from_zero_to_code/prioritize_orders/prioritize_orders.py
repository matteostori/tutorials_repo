import pandas as pd

# 1. Create the total orders list by merging the two Excel files (”Orders_1.xlsx” and “Orders_2.xlsx”)
#   1.1 Open all input excel and prepare data (e.g. clean up, harmonize columns)

#------> open all input files       
print('Opening all input Excels: orders, customer list, inventory status...')
df_orders_1 = pd.read_excel('../tutorials_repo/data_sources/microelectronics_products/Orders_1.xlsx')
df_orders_2 = pd.read_excel('../tutorials_repo/data_sources/microelectronics_products/Orders_2.xlsx')
df_customers = pd.read_excel('../tutorials_repo/data_sources/microelectronics_products/Customers_list.xlsx')
df_inventory = pd.read_excel('../tutorials_repo/data_sources/microelectronics_products/Inventory_status.xlsx')
print('Input Excels opened')

#------> print some stats (row numbers of each files)
print('Orders_1 file contains {} rows'.format(len(df_orders_1)))
print('Orders_2 file contains {} rows'.format(len(df_orders_2)))
print('Customer list file contains {} rows'.format(len(df_customers)))
print('Inventory file contains {} rows'.format(len(df_inventory)))

#------> clean up Order 1 data
df_orders_1 = df_orders_1[df_orders_1['Order line'].notna()] #second row is empty and have to be deleted
print('Orders_1 file contains {} rows'.format(len(df_orders_1)))

#------> harmonize Order 1 columns
df_orders_1 = df_orders_1.rename(columns={'Total €': 'Total eur', 'Unit price (€ / piece)': 'Price eur / piece',
                                'Volume (# pieces)': '# pieces', 'SKU code': 'SKU'})
#------> include additional columns in Order 1
df_orders_1['source'] = 'Orders_1.xlsx' #add one column to include the source of the information 
df_orders_1['Notes'] = '' #add an empty column 'Notes'

#------> clean up Order 2 data
df_orders_2 = df_orders_2[df_orders_2['Order line'].notna()] #second row is empty and have to be deleted
print('Orders_2 file contains {} rows'.format(len(df_orders_2)))
df_orders_2['source'] = 'Orders_2.xlsx' #add one column to include the source of the information

#------> harmonize Order 2 columns
df_orders_2 = df_orders_2.rename(columns={'Price': 'Price eur / piece', 'Total': 'Total eur'})
df_orders_2['Sales rep'] = '' #add an empty column 'Sales rep'
df_orders_2['Order status'] = 0 #set order status to 0 (i.e. not confirmed)

#------> check both dataframes contain the same columns, by printing them
print(df_orders_1.columns)
print(df_orders_2.columns)

#   1.2 Create a single dataframe containing both Order 1 and Order 2 input
df_orders_concat_list = [df_orders_1, df_orders_2]
df_orders = pd.concat(df_orders_concat_list)
#------> check the newly created dataframe
print(df_orders)
print('Orders dataframe contains {} rows and columns'.format(df_orders.shape))

#   1.3 Include “Sales rep”, “Priority” and “Country”, pulling them from the Customer list by matching 
#       the customer code in the order (customer code is the 'key' for this match)
df_orders = df_orders.merge(df_customers, on='Customer code', how='left')
print(df_orders)

# 2. Confirm orders based on Customer priority, requested delivery date, subject to availability in inventory
#   2.1. Create a "pivot table" to compare total # of pieces ordered by SKU and corresponding inventory 

#------> create a new dataframe containing only the total amount of pieces ordered by SKU
#------> the resulting datafame has only two columns: 'SKU' and '# pieces'
#------> this is the equivalent of creating a pivot table in excel and drag&dropping 'SKU' in the rows and '# pieces' in sum of values boxes
df_orders_pivot = df_orders.groupby('SKU')[['SKU', '# pieces']].sum()

#------> rename the inventory column to avoid confusion with the '# pieces' from the orders
df_inventory = df_inventory.rename(columns={'# pieces': '# pieces in inventory'})
#------> pull the information of the inventory from the Inventory_status file, as already sone for the Customers list
df_orders_pivot = df_orders_pivot.merge(df_inventory, on='SKU', how='left')
df_orders_pivot.loc[df_orders_pivot['# pieces in inventory'] == 'out of stock', '# pieces in inventory'] = -1
#------> create a new column 'Inventory sufficient' to check if we have enough inventory for each SKU
df_orders_pivot['Inventory sufficient'] = df_orders_pivot['# pieces'] <= df_orders_pivot['# pieces in inventory'] 

#   2.2 Include the information of which order can be flagged as 'confirmed' (inventory levels >= quantities ordered)

#------> merge the df_order_pivot in the df_order database; this will include the new column 'Inventory sufficient'
#------> to check if an SKU has sufficient pieces in inventory, and the order can be confirmed
df_orders_pivot = df_orders_pivot.rename(columns={'# pieces': '# pieces by SKU'})
df_orders = df_orders.merge(df_orders_pivot, on='SKU', how='left')

#------> get rid of duplicated / unused columns
df_orders = df_orders.drop(['Sales rep_x', '# pieces by SKU'], axis=1) #axis = 1 means we want to drop a columns, axis = 0 would drop a row 

#   2.3 Change the “Order status” column to “1” (assuming 1 = confirmed) to each SKU having sufficient inventory
df_orders.loc[df_orders['Inventory sufficient'] == True, 'Order status'] = 1

#   2.4 For each product, check if ordered quantity <= inventory level; in this case change the “Order status” column to “1” 
#       (assuming 1 = confirmed) and decrease the inventory levels by the corresponding quantity

#------> sort the dataframe by priority (i.e. we want on top all rows with Priority 0 - the highest, then Priority 1 and 2)
#------> and then by Customer requested date
df_orders = df_orders.sort_values(by=['Priority', 'Customer requested date'])

df_orders.reset_index(drop=True, inplace=True) #reset the index of the dataframe
#------> iterate each row of the Orders dataframe
for i in df_orders.index:
    sku = df_orders['SKU'][i] #get the SKU for this order
    remaining_inventory_df = df_orders_pivot.loc[df_orders_pivot['SKU'] == sku] #get the current inventory level
    remaining_inventory = remaining_inventory_df['# pieces in inventory'].iloc[0]
    #if the current inventory level is >= ordered quantity, then confirm the order
	#and decrease the inventory level by the same amount
    if df_orders['# pieces'][i] <= remaining_inventory:
        df_orders.loc[i, 'Order status'] = 1
        delta_inventory = remaining_inventory - df_orders['# pieces'][i]
        df_orders_pivot.loc[df_orders_pivot['SKU'] == sku, '# pieces in inventory'] = delta_inventory

# 3.  Print the data in Excel format and analyze the results

df_orders.to_excel('python/tutorials/from_zero_to_code/prioritize_orders/output/Orders.xlsx') #Excel with the Orders
df_orders_pivot.to_excel('python/tutorials/from_zero_to_code/prioritize_orders/output/Inventory_final.xlsx') #File with final inventory