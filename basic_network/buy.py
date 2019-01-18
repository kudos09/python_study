import sys
import copy


class Product:
    def __init__(self, pid, name, price, count):
        self.pid = pid
        self.name = name
        self.price = price
        self.count = count

    def __str__(self):
        return "id>>" + str(self.pid) + "  \tname>>" + self.name+"  \tprice>>" + str(self.price) + " \tcount>>" + str(self.count)


class ProductList:
    # 输入列表，抓成以id为key的字典，便于以后查找
    def __init__(self, products=[]):
        self.prod_list = dict()
        for product in products:
            self.prod_list[product.pid] = product

    def minus(self, productId, num):
        if productId in self.prod_list:
            if self.prod_list[productId].count > num:
                self.prod_list[productId].count -= num
                return True
            elif self.prod_list[productId].count == num:
                self.prod_list.pop(productId)
                return True
            else:
                print("库存不够，请重新选择数量")
                return False
        else:
            print("本仓库没有此件商品，请重新选择")

    def add(self, product, num=1):
        if product.pid in self.prod_list:
            self.prod_list[product.pid].count += num
        else:
            new_product = copy.deepcopy(product)
            new_product.count = num
            self.prod_list[new_product.pid] = new_product

    def print_all_product(self):
        for key in self.prod_list:
            print(self.prod_list[key])


class Shopping:
    @classmethod
    def get_salary(cls):
        salary = input("请输入您购买卡的余额：")
        if not salary.isdecimal():
            print("您输入的卡的余额不是数值型，请重新输入 ")
            return Shopping.get_salary()
        else:
            return int(salary)

    def __init__(self, sell_product=[]):
        self.salary = Shopping.get_salary()
        self.sell_product = ProductList(sell_product)
        self.buy_product = ProductList()

    def buy_some_product(self):
        self.sell_product.print_all_product()
        select_product = input("请输入您的产品编码： ")
        product_num = input("请输入商品的数量： ")
        if product_num.isdigit():
            product_num = int(product_num)
        else:
            print("数量必须是数字！")
            self.buy_some_product()
        # 如果够支付，则工资减少，同时buy_product增加一个商品
        if self.is_afford(select_product, product_num):
            self.buy_product.add(self.sell_product.prod_list[select_product], product_num)
            self.salary -= self.sell_product.prod_list[select_product].price*product_num
            self.sell_product.minus(select_product, product_num)
            print("您当前购买商品如下：")
            self.buy_product.print_all_product()
            print("您的余额是 %s 元" % self.salary)
        else:
            print("您的余额不足，请重新选择")

    # 判断当前的工资余额是否够支付产品
    def is_afford(self, procudtId, product_num):
        if type(procudtId) == int:
            procudtId = str(procudtId)
        if self.salary >= self.sell_product.prod_list[procudtId].price * product_num:
            return True
        else:
            return False

    def select_process(self):
        while True:
            selector = input("""
                请输入您的选择？
                    购买(b)
                    退出(q)
                """)
            if selector == 'b':
                self.buy_some_product()
            else:
                print("您购买了如下商品")
                self.buy_product.print_all_product()
                print("您的余额是 %s 元" % self.salary)
                print("欢迎您下次光临")
                sys.exit(0)



if __name__ == '__main__':
    # 测试初始化产品列表的信息的功能
    init_product=[Product('1', 'bike', 100, 10), Product('2', 'ipad', 3000, 3), Product('3', 'paper', 500, 30)]
    shop = Shopping(init_product)
    shop.select_process()