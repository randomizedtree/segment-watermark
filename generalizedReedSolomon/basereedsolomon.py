import galois as gl
from generalizedReedSolomon import helper

class Base_Reed_Solomon():
    def __init__(self, field_size:int, message_length:int, payload_length:int,symbol_size:int,multi_processing,irr_poly=None,debug=True,p=1,):
        self.field_size = field_size
        if multi_processing:
            self.pool = Pool(p)
        else:
            self.pool = None
        if irr_poly ==None:
            try:
                self.galois_field = gl.GF(field_size**symbol_size)#TODO remove power asap
            except LookupError:
                print("need to find irreducible polynominal manually")
                irreducible_polynominal = gl.irreducible_poly(field_size,symbol_size)
                self.galois_field = gl.GF(field_size**symbol_size,irreducible_poly=irreducible_polynominal)   
        else:
            self.galois_field = gl.GF(field_size**symbol_size,irreducible_poly=irr_poly)   
        self.message_length = message_length
        self.payload_length = payload_length
        self.two_s = message_length - payload_length
        self.helper = helper.Galois_Helper(self.galois_field,self.pool,debug)
        self.multi = multi_processing
    

    def convert_to_symbol_array(self,array):
        
        if self.galois_field.degree > 1:
            output = []
            for i in range(0,len(array)):
                output.append(int(self.galois_field.Vector(array[i])))
            return output
        return array
    
    def symbol_array_to_array(self,array):
        if self.galois_field.degree > 1:
            output = []
            for i in range(0,len(array)):
                output.append(self.galois_field(array[i]).vector().tolist())
            return output
        return array
