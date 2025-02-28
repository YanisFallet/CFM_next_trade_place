def compute_spreads(data):
    for order_book in range(6):
        data[f"spread_{order_book}"] = data[(order_book, 'ask')] - data[(order_book, "bid")]
    return data

def mean_qty_past_orders(data, on_all_data=False):
    if not on_all_data:
        data["mean_qty_past_orders"] = data[[(i, "qty") for i in range(10)]].mean(axis=1)
        return data
    else:
        data_mean = data.groupby("stock_id")[[(i, "qty") for i in range(10)]].mean().mean(axis=1).rename("mean_qty_past_orders")
        return data.join(data_mean, on="stock_id")