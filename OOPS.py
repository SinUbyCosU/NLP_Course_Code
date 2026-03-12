#Classes and objects
class Car:
    def __init__(self,brand,color):
        self.brand=brand
        self.color=color
    
    def drive(self):
        print(f"The {self.color} {self.brand} is driving..")

my_car=Car("BMW","Black")

my_car.drive()

class BankAccount:
    def __init__(self,initial_balance):
        self.__balance=initial_balance
    def get_balance(self):
        return self.__balance

    def deposit(self,amount):
        if amount>0:
            self.__balance +=amount

account=BankAccount(1000000)
print(account.get_balance())
account.deposit(500000)
print(account.get_balance())


#Inheritance

class Animal:
    def eat(self):
        print("animal is eating")

class Dog(Animal):
    def bark(self):
        print("woof")

my_dog=Dog()
my_dog.eat()
my_dog.bark()

class Cat:
    def make_Sound(self):
        return "meow"
class Duck:
    def make_Sound(self):
        return "quack"
def play_sound(animal_object):
    print(animal_object.make_Sound())

play_sound(Cat())
play_sound(Duck())