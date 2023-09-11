class Employee:
    def __init__(self,name,email,age,location):
        self.name=name
        self.email=email
        self.age=age
        self.location=location

    def display(self):
        return f"{self.name} {self.email} {self.age} {self.location}"
    

maurice=Employee('Maurice Oboya','oboyamaurice@gmail.com',25,'Nairobi')
kim=Employee('Kimani ichungwa','kimani3@outlook.com',21,'Tarakanithi')


print(maurice.display())
print(kim.display())