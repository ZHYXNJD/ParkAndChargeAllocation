class Parkinglot:
    def __init__(self, id, total_num, charge_num, slow_charge_num, park_fee=6, fast_charge_fee=0.7, slow_charge_fee=0.5,
                 reserve_fee=1):  # 初始化一个停车场，需要定义停车场编号，泊位数量，充电桩比例
        self.id = id
        self.total_num = total_num
        self.charge_num = charge_num
        self.ordinary_num = total_num - charge_num
        self.slow_charge_space = slow_charge_num
        self.fast_charge_space = charge_num - slow_charge_num
        self.park_fee = park_fee
        self.fast_charge_fee = fast_charge_fee
        self.slow_charge_fee = slow_charge_fee
        self.reserve_fee = reserve_fee


def get_parking_lot(parking_lot_num=4,config='1:1'):

    if config == '1:1':
        pl1 = Parkinglot(id=1, total_num=40, charge_num=20, slow_charge_num=4)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=10, slow_charge_num=2)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=5, slow_charge_num=1)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=15, slow_charge_num=3)
    elif config == '2:1':
        pl1 = Parkinglot(id=1, total_num=40, charge_num=13, slow_charge_num=2)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=7, slow_charge_num=1)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=3, slow_charge_num=1)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=10, slow_charge_num=2)
    elif config == '1:2':
        pl1 = Parkinglot(id=1, total_num=40, charge_num=27, slow_charge_num=5)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=13, slow_charge_num=3)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=7, slow_charge_num=1)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=20, slow_charge_num=4)
    elif config == '3:1':
        pl1 = Parkinglot(id=1, total_num=40, charge_num=10, slow_charge_num=3)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=5, slow_charge_num=1)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=2, slow_charge_num=1)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=8, slow_charge_num=2)
    elif config == '1:3':
        pl1 = Parkinglot(id=1, total_num=40, charge_num=30, slow_charge_num=6)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=15, slow_charge_num=3)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=8, slow_charge_num=2)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=22, slow_charge_num=4)
    elif config == 0.1:
        pl1 = Parkinglot(id=1, total_num=40, charge_num=36, slow_charge_num=7)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=18, slow_charge_num=3)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=9, slow_charge_num=2)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=27, slow_charge_num=6)
    elif config == 0.2:
        pl1 = Parkinglot(id=1, total_num=40, charge_num=32, slow_charge_num=6)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=16, slow_charge_num=3)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=8, slow_charge_num=2)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=24, slow_charge_num=5)
    elif config == 0.3:
        pl1 = Parkinglot(id=1, total_num=40, charge_num=28, slow_charge_num=5)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=14, slow_charge_num=3)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=7, slow_charge_num=2)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=21, slow_charge_num=4)
    elif config == 0.4:
        pl1 = Parkinglot(id=1, total_num=40, charge_num=24, slow_charge_num=5)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=12, slow_charge_num=3)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=6, slow_charge_num=1)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=18, slow_charge_num=3)
    elif config == 0.5:
        pl1 = Parkinglot(id=1, total_num=40, charge_num=20, slow_charge_num=4)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=10, slow_charge_num=2)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=5, slow_charge_num=1)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=15, slow_charge_num=3)
    elif config == 0.6:
        pl1 = Parkinglot(id=1, total_num=40, charge_num=16, slow_charge_num=4)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=8, slow_charge_num=1)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=4, slow_charge_num=1)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=12, slow_charge_num=2)
    elif config == 0.7:
        pl1 = Parkinglot(id=1, total_num=40, charge_num=12, slow_charge_num=3)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=6, slow_charge_num=1)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=3, slow_charge_num=1)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=9, slow_charge_num=1)
    elif config == 0.8:
        pl1 = Parkinglot(id=1, total_num=40, charge_num=8, slow_charge_num=1)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=4, slow_charge_num=1)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=2, slow_charge_num=1)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=6, slow_charge_num=1)
    elif config == 0.9:
        pl1 = Parkinglot(id=1, total_num=40, charge_num=4, slow_charge_num=1)
        pl2 = Parkinglot(id=2, total_num=20, charge_num=2, slow_charge_num=0)
        pl3 = Parkinglot(id=3, total_num=10, charge_num=1, slow_charge_num=0)
        pl4 = Parkinglot(id=4, total_num=30, charge_num=3, slow_charge_num=1)
    else:
        print("需要先确定配建比例！")
        return None

    # 4个停车场
    # pl1 = Parkinglot(id=1, total_num=40, charge_num=10, slow_charge_num=3)
    # pl2 = Parkinglot(id=2, total_num=20, charge_num=5, slow_charge_num=1)
    # pl3 = Parkinglot(id=3, total_num=10, charge_num=2, slow_charge_num=1)
    # pl4 = Parkinglot(id=4, total_num=30, charge_num=8, slow_charge_num=2)
    # 1:1 配建 （快充:慢充 4:1）
    # pl1 = Parkinglot(id=1, total_num=40, charge_num=32, slow_charge_num=6)
    # pl2 = Parkinglot(id=2, total_num=20, charge_num=16, slow_charge_num=3)
    # pl3 = Parkinglot(id=3, total_num=10, charge_num=8, slow_charge_num=2)
    # pl4 = Parkinglot(id=4, total_num=30, charge_num=24, slow_charge_num=5)

    # pl1 = Parkinglot(id=1, total_num=40, charge_num=20, slow_charge_num=4)
    # pl2 = Parkinglot(id=2, total_num=20, charge_num=10, slow_charge_num=2)
    # pl3 = Parkinglot(id=3, total_num=10, charge_num=5, slow_charge_num=1)
    # pl4 = Parkinglot(id=4, total_num=30, charge_num=15, slow_charge_num=3)

    # pl1 = Parkinglot(id=1, total_num=40, charge_num=8, slow_charge_num=1)
    # pl2 = Parkinglot(id=2, total_num=20, charge_num=4, slow_charge_num=1)
    # pl3 = Parkinglot(id=3, total_num=10, charge_num=2, slow_charge_num=1)
    # pl4 = Parkinglot(id=4, total_num=30, charge_num=6, slow_charge_num=1)

    # pl1 = Parkinglot(id=1, total_num=40, charge_num=10, slow_charge_num=2)
    # pl2 = Parkinglot(id=2, total_num=20, charge_num=5, slow_charge_num=1)
    # pl3 = Parkinglot(id=3, total_num=10, charge_num=4, slow_charge_num=1)
    # pl4 = Parkinglot(id=4, total_num=30, charge_num=6, slow_charge_num=1)

    pl = [pl1, pl2, pl3, pl4]
    current_pl_num = len(pl)
    if current_pl_num < parking_lot_num:
        print(f"目前已初始化{current_pl_num}个停车场，还需要新添加{parking_lot_num - current_pl_num}个停车场")
    elif current_pl_num > parking_lot_num:
        return pl[:parking_lot_num]
    else:
        return pl
