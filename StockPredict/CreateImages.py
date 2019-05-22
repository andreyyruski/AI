import os
import pandas as pd
import matplotlib.pyplot as plt
#from concurrent import futures

data_folder = 'data'
image_folder = 'images'

def create_images(ticker):
    
    print('Creating ' + ticker)

    ticker_data_df = pd.read_csv(data_folder + '/' + ticker + '_data.csv')

    #print(ticker_data_df.info())
    #print(ticker_data_df.describe())

    # add percentage column
    ticker_data_df['percent'] = (ticker_data_df['close'] / ticker_data_df['open']) - 1

    # chart
    # ticker_data_df.plot(x = 'date', y = 'close')
    # ticker_data_df.plot(x = 'date', y = 'percent')
    # plt.show()

    # Converting 'date' row to DatetimeIndex for Grouper to run
    ticker_data_df['date'] = pd.to_datetime(ticker_data_df['date'])
    ticker_data_df = ticker_data_df.set_index('date')

    # splitting to chuncks of weeks
    grp = ticker_data_df.groupby(pd.Grouper(freq='W'))
    weeks = [g for n, g in grp]

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    amount = len(weeks)
    for i in range(0, amount -1):
        
        # determine the class 0 is < 5%, 1 is < 2.5%, 3 is 0%, 4 is > 2.5%, 5 is > 5% 
        next_week_open = weeks[i+1]['open'][0]
        next_week_close = next_week_open
        days = weeks[i+1]['close'].count()
        if days > 0:
            next_week_close = weeks[i+1]['close'][days - 1]
            
        next_week_percent =  (next_week_close / next_week_open) - 1
        if next_week_percent < -0.05:
            classification = 0 # high loss
        elif next_week_percent < -0.025:
            classification = 1 # light loss
        elif next_week_percent < 0:
            classification = 2 # very light loss
        elif next_week_percent < 0.025:
            classification = 3 # very light gain
        elif next_week_percent < 0.05:
            classification = 4 # light gain
        else:
            classification = 5 # high gain
            
        #print(str(round(next_week_percent * 100, 2)) + '% = Class #' + str(classification))
        
        axes = weeks[i].plot(y = 'percent')
        axes.set_ylim(-0.05, 0.05) # range -5% to 5%
        axes.axis('off')
        axes.get_legend().remove()
        
        # create folder if needed
        class_folder = image_folder + '/' + str(classification)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
            
        # save image
        fig = axes.get_figure()
        fig.savefig(class_folder + '/' + ticker + '_' + str(i) + '.png')
        #fig.set_visible(False)
        plt.close(fig)

if __name__ == '__main__':
    

    """ list of s_anp_p companies """
    s_and_p = ['MMM','ABT','ABBV','ACN','ATVI','AYI','ADBE','AMD','AAP','AES','AET',
        'AMG','AFL','A','APD','AKAM','ALK','ALB','ARE','ALXN','ALGN','ALLE',
        'AGN','ADS','LNT','ALL','GOOGL','GOOG','MO','AMZN','AEE','AAL','AEP',
        'AXP','AIG','AMT','AWK','AMP','ABC','AME','AMGN','APH','APC','ADI','ANDV',
        'ANSS','ANTM','AON','AOS','APA','AIV','AAPL','AMAT','APTV','ADM','ARNC',
        'AJG','AIZ','T','ADSK','ADP','AZO','AVB','AVY','BHGE','BLL','BAC','BK',
        'BAX','BBT','BDX','BRK.B','BBY','BIIB','BLK','HRB','BA','BWA','BXP','BSX',
        'BHF','BMY','AVGO','BF.B','CHRW','CA','COG','CDNS','CPB','COF','CAH','CBOE',
        'KMX','CCL','CAT','CBG','CBS','CELG','CNC','CNP','CTL','CERN','CF','SCHW',
        'CHTR','CHK','CVX','CMG','CB','CHD','CI','XEC','CINF','CTAS','CSCO','C','CFG',
        'CTXS','CLX','CME','CMS','KO','CTSH','CL','CMCSA','CMA','CAG','CXO','COP',
        'ED','STZ','COO','GLW','COST','COTY','CCI','CSRA','CSX','CMI','CVS','DHI',
        'DHR','DRI','DVA','DE','DAL','XRAY','DVN','DLR','DFS','DISCA','DISCK','DISH',
        'DG','DLTR','D','DOV','DWDP','DPS','DTE','DRE','DUK','DXC','ETFC','EMN','ETN',
        'EBAY','ECL','EIX','EW','EA','EMR','ETR','EVHC','EOG','EQT','EFX','EQIX','EQR',
        'ESS','EL','ES','RE','EXC','EXPE','EXPD','ESRX','EXR','XOM','FFIV','FB','FAST',
        'FRT','FDX','FIS','FITB','FE','FISV','FLIR','FLS','FLR','FMC','FL','F','FTV',
        'FBHS','BEN','FCX','GPS','GRMN','IT','GD','GE','GGP','GIS','GM','GPC','GILD',
        'GPN','GS','GT','GWW','HAL','HBI','HOG','HRS','HIG','HAS','HCA','HCP','HP','HSIC',
        'HSY','HES','HPE','HLT','HOLX','HD','HON','HRL','HST','HPQ','HUM','HBAN','HII',
        'IDXX','INFO','ITW','ILMN','IR','INTC','ICE','IBM','INCY','IP','IPG','IFF','INTU',
        'ISRG','IVZ','IQV','IRM','JEC','JBHT','SJM','JNJ','JCI','JPM','JNPR','KSU','K','KEY',
        'KMB','KIM','KMI','KLAC','KSS','KHC','KR','LB','LLL','LH','LRCX','LEG','LEN','LUK',
        'LLY','LNC','LKQ','LMT','L','LOW','LYB','MTB','MAC','M','MRO','MPC','MAR','MMC','MLM',
        'MAS','MA','MAT','MKC','MCD','MCK','MDT','MRK','MET','MTD','MGM','KORS','MCHP','MU',
        'MSFT','MAA','MHK','TAP','MDLZ','MON','MNST','MCO','MS','MOS','MSI','MYL','NDAQ',
        'NOV','NAVI','NTAP','NFLX','NWL','NFX','NEM','NWSA','NWS','NEE','NLSN','NKE','NI',
        'NBL','JWN','NSC','NTRS','NOC','NCLH','NRG','NUE','NVDA','ORLY','OXY','OMC','OKE',
        'ORCL','PCAR','PKG','PH','PDCO','PAYX','PYPL','PNR','PBCT','PEP','PKI','PRGO','PFE',
        'PCG','PM','PSX','PNW','PXD','PNC','RL','PPG','PPL','PX','PCLN','PFG','PG','PGR',
        'PLD','PRU','PEG','PSA','PHM','PVH','QRVO','PWR','QCOM','DGX','RRC','RJF','RTN','O',
        'RHT','REG','REGN','RF','RSG','RMD','RHI','ROK','COL','ROP','ROST','RCL','CRM','SBAC',
        'SCG','SLB','SNI','STX','SEE','SRE','SHW','SIG','SPG','SWKS','SLG','SNA','SO','LUV',
        'SPGI','SWK','SBUX','STT','SRCL','SYK','STI','SYMC','SYF','SNPS','SYY','TROW','TPR',
        'TGT','TEL','FTI','TXN','TXT','TMO','TIF','TWX','TJX','TMK','TSS','TSCO','TDG','TRV',
        'TRIP','FOXA','FOX','TSN','UDR','ULTA','USB','UAA','UA','UNP','UAL','UNH','UPS','URI',
        'UTX','UHS','UNM','VFC','VLO','VAR','VTR','VRSN','VRSK','VZ','VRTX','VIAB','V','VNO',
        'VMC','WMT','WBA','DIS','WM','WAT','WEC','WFC','HCN','WDC','WU','WRK','WY','WHR','WMB',
        'WLTW','WYN','WYNN','XEL','XRX','XLNX','XL','XYL','YUM','ZBH','ZION','ZTS']
    

    # workers = 10
    # with futures.ThreadPoolExecutor(workers) as executor:
    #     res = executor.map(create_images, s_and_p)

    for ticker in s_and_p:
        create_images(ticker)

    print('-----------------> Done <-----------------')
