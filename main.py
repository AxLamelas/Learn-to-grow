import tensorflow as tf
import pickle


'''
Notacao
t_x = comprimento de x
n_x = numero de x's
f_x = forma de x (exemplo: 2x3)
c_x = camada x
'''

#Importacao das imagens e legendas
#Imagens pequenas

with open('Info/p_imgs_treino.pckl', 'rb') as f:
    p_imgs_treino = pickle.load(f)
    
    
    
with open('Info/p_imgs_teste.pckl', 'rb') as f:
    p_imgs_teste = pickle.load(f)
    
with open('Info/p_imgs_validacao.pckl', 'rb') as f:
    p_imgs_validacao = pickle.load(f)
    
    
#Imagens originais    
    
with open('Info/o_imgs_treino.pckl', 'rb') as f:
    o_imgs_treino = pickle.load(f)
    
with open('Info/o_imgs_teste.pckl', 'rb') as f:
    o_imgs_teste = pickle.load(f)
    
with open('Info/o_imgs_validacao.pckl', 'rb') as f:
    o_imgs_validacao = pickle.load(f)

#Legendas
    
with open('Info/l_treino.pckl', 'rb') as f:
    l_treino = pickle.load(f)
    
with open('Info/l_teste.pckl', 'rb') as f:
    l_teste = pickle.load(f)
    
with open('Info/l_validacao.pckl', 'rb') as f:
    l_validacao = pickle.load(f)



t_img = 14

n_canais = 1 #Imagens usadas a preto e branco
n_classes = 10 #Possiveis classificacoes, neste caso 10 porque as imagens utilizadas sao 0, 1, 2 ,...,9






#Variaveis
t_lote_treino = 64
t_lote_teste = 256
iter_total = 0
lote_atual = 0
    
def sumario_varivel(var):
#Para visualizacao dos parametros
    with tf.name_scope('sumarios'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('media', mean)
    with tf.name_scope('des_padrao'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('des_padrao', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histograma', var)    

def novos_pesos(forma):
    return tf.Variable(tf.truncated_normal(forma, stddev = 0.05))

def novos_deslocamentos(comp):
    return tf.Variable(tf.constant(0.05, shape = [comp]))

def nova_c_conv_trans(entrada,t_img,escala,n_canais_entrada,t_filtro,n_canais_saida,salto = 1, nome='conv_trans'):
    with tf.name_scope(nome):
        forma = [t_img,t_img,n_canais_saida,n_canais_entrada]        
        with tf.name_scope('pesos'):
            pesos = novos_pesos(forma = forma)
            sumario_varivel(pesos)
       
        with tf.name_scope('ant_ativacao'):
            forma_saida = [-1,escala*t_img,escala*t_img,n_canais_saida]
            camada = tf.nn.conv2d_transpose(value = entrada, filter = pesos,output_shape = forma_saida, strides = [1, salto, salto, 1], padding = 'SAME')
            sumario_varivel(camada)

        with tf.name_scope('ativacao'):       
            camada = tf.nn.relu(camada)
            sumario_varivel(camada)
    
        return camada, pesos


def nova_c_conv(entrada,n_canais,t_filtro,n_filtros, amostragem = False,salto = 1, nome='conv'):
    with tf.name_scope(nome):
        forma = [t_filtro, t_filtro, n_canais, n_filtros]
        with tf.name_scope('pesos'):
            pesos = novos_pesos(forma = forma)
            sumario_varivel(pesos)
        with tf.name_scope('deslocamentos'):
            deslocamentos = novos_deslocamentos(comp = n_filtros)
            sumario_varivel(deslocamentos)
        with tf.name_scope('ant_ativacao'):
            camada = tf.nn.conv2d(input = entrada, filter = pesos, strides = [1, salto, salto, 1], padding = 'SAME')
            camada += deslocamentos
            sumario_varivel(camada)
    
        if amostragem == True:
            with tf.name_scope('amostragem'):
                camada = tf.nn.max_pool(value=camada,ksize=[1,2,2,1],strides = [1,2,2,1], padding = 'SAME')
                sumario_varivel(camada)
                
        with tf.name_scope('ativacao'):       
            camada = tf.nn.relu(camada)
            sumario_varivel(camada)
    
        return camada, pesos

def camada_para_martis(camada):
    f_camada = camada.shape
    n_caracteristicas = f_camada[1:4].num_elements()
    martis_camada = tf.reshape(camada, [-1, n_caracteristicas])

    return martis_camada, n_caracteristicas

def nova_camada_conectada(entrada, n_inputs, n_outputs, relu = True, nome = 'tot_conectada'):
    
    with tf.name_scope(nome):
        with tf.name_scope('pesos'):
            pesos = novos_pesos(forma = [n_inputs, n_outputs])
            sumario_varivel(pesos)
            
        with tf.name_scope('deslocamentos'):
            deslocamentos = novos_deslocamentos(comp = n_outputs)
            sumario_varivel(deslocamentos)
        with tf.name_scope('ant_ativacao'):
            camada = tf.matmul(entrada, pesos) + deslocamentos
            sumario_varivel(camada)

        if relu:
            with tf.name_scope('ativacao'):
                camada = tf.nn.relu(camada)
                sumario_varivel(camada)
        
        return camada


escala = 2
#Rede neural convolucional
#Camada 1
t_filtro_1 = 2
n_filtro_1 = 25


#Camada 2
t_filtro_2 = 2
n_filtro_2 = 50



#Camada totalmente conectada

n_conectada_1 = 128
 


sessao = tf.Session(config=tf.ConfigProto(log_device_placement=True))


with tf.name_scope('entrada'):
    x = tf.placeholder(tf.float32, shape = [None, t_img*t_img], name='x')
    x_img = tf.reshape(x, [-1, t_img, t_img, n_canais]) # [n_img,t_img,t_img,n_canais] tensor com os valores de cada pixel de cada imagem
    tf.summary.image('x', x_img,1)
    y = tf.placeholder(tf.float32, shape = [None, escala**2*t_img*t_img], name='y')
    y_real = tf.reshape(y, [-1, escala*t_img, escala*t_img, n_canais])
    tf.summary.image('y_real', y_real,1)
    vec_classe_real = tf.placeholder(tf.float32, shape=[None, 10], name='vec_classe_real')
    classe_real = tf.argmax(vec_classe_real, dimension=1)
    tf.summary.histogram('classe_real', classe_real)




c_conv_trans, pesos_conv_trans = nova_c_conv_trans(x_img,t_img,escala,n_canais,t_filtro_1,n_canais)

tf.summary.image('resultado',c_conv_trans)

c_conv_1, pesos_conv_1 = nova_c_conv(entrada= x_img, n_canais = n_canais, t_filtro = t_filtro_1, n_filtros = n_filtro_1,amostragem = True, nome = 'conv3')
c_conv_2, pesos_conv_2 = nova_c_conv(entrada = c_conv_1, n_canais = n_filtro_1, t_filtro = t_filtro_2, n_filtros = n_filtro_2,amostragem = True, nome ='conv2')
matris_camada, n_caracteristicas = camada_para_martis(c_conv_2)

c_conectada_1 = nova_camada_conectada(entrada = matris_camada,n_inputs = n_caracteristicas, n_outputs = n_conectada_1, relu = True)
c_conectada_2 = nova_camada_conectada(entrada = c_conectada_1,n_inputs = n_conectada_1, n_outputs = n_classes, relu = False)


vec_classe_prevista = tf.nn.softmax(c_conectada_2)
classe_prevista = tf.argmax(vec_classe_prevista, dimension=1)


#Funcao de erro

peso_classi = tf.Variable(1.0)
peso_img = tf.Variable(1.0)

with tf.name_scope('erro'):
    with tf.name_scope('entropia_cruzada'):
        entropia_cruzada = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=c_conectada_2, labels = vec_classe_real))
        tf.summary.scalar('entropia cruzada', entropia_cruzada)
    with tf.name_scope('mdq'):
        mdq = tf.reduce_mean(tf.squared_difference(c_conv_trans, y_real, name = 'mdq'))
        tf.summary.scalar('mdq',mdq)
        
    erro = tf.square(tf.add(tf.multiply(peso_img,mdq), tf.multiply(peso_classi, entropia_cruzada)),name='sErros')
    tf.summary.scalar('erro',erro)
    tf.summary.scalar('peso_classi',peso_classi)

#Optimizacao pelo metodo do gradiente
with tf.name_scope('optimizacao'):
    optimizador = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(erro)
    
with tf.name_scope('precisao'):
    with tf.name_scope('precisao_classe'):
        precisao_classe = tf.reduce_mean(tf.cast(tf.equal(classe_prevista, classe_real), tf.float32))
    with tf.name_scope('precisao_pixeis'):
        precisao_pixeis = tf.reduce_mean(tf.cast(tf.equal(c_conv_trans, y_real), tf.float32))
tf.summary.scalar('precisao_classe', precisao_classe)
tf.summary.scalar('precisao_pixeis',precisao_pixeis)
   

sumarios = tf.summary.merge_all()    
sumarios_treino = tf.summary.FileWriter('treino')
sumarios_treino.add_graph(sessao.graph)
sumarios_teste = tf.summary.FileWriter('teste')

sessao.run(tf.global_variables_initializer())
gravador = tf.train.Saver()








pos_atual_teste = 0
pos_atual_treino  = 0

def dic_entrada(treino):    
    global pos_atual_teste 
    global pos_atual_treino
    
       
    if treino:
        
        pos_atual_treino += t_lote_treino 
        x_lote = p_imgs_treino[pos_atual_treino-t_lote_treino:pos_atual_treino ]
        y_real_lote = o_imgs_treino[pos_atual_treino-t_lote_treino:pos_atual_treino ]
        vec_classe_real_lote = l_treino[pos_atual_treino-t_lote_treino:pos_atual_treino ]           
        
        dic = {x: x_lote, vec_classe_real: vec_classe_real_lote, y: y_real_lote }
        
    else:
        pos_atual_teste += t_lote_teste 
        x_lote = p_imgs_teste[pos_atual_teste-t_lote_teste:pos_atual_teste ]
        y_real_lote = o_imgs_teste[pos_atual_teste-t_lote_teste:pos_atual_teste ]
        vec_classe_real_lote = l_teste[pos_atual_teste-t_lote_teste:pos_atual_teste ]           
        
        dic = {x: x_lote, vec_classe_real: vec_classe_real_lote, y: y_real_lote }
        
                
    
    return dic

    
max_treino = len(p_imgs_treino)
max_teste = len(p_imgs_teste)


peso_img.assign(tf.cast(0.01,tf.float32))

#Apenas 10% das imagens por causa de limitacoes de tempo para treinar
for i in range(int(max_treino)):
    sumario, _ = sessao.run([sumarios, optimizador],dic_entrada(True))
    sumarios_treino.add_summary(sumario,i)
    print(i)
    if i%500 == 0:
        gravador.save(sessao, './modelo',i)


for i in range(int(max_teste)):
    sumario = sessao.run(sumarios, dic_entrada(False))
    sumarios_teste.add_summary(sumario,i)   
    
        
        
        

