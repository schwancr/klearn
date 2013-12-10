
from msmbuilder.metrics.parsers import add_argument, add_basic_metric_parsers, construct_basic_metric
from ktica.kernels import DotProduct, Gaussian, Polynomial

def add_kernel_parsers(parser):
    
    kernel_parser_list = []
    kernel_subparser = parser.add_subparsers(dest='kernel', description='Available kernels to use.')

    kernel_parser_list.extend( add_layer_kernel_parsers(kernel_subparser) )

    return kernel_parser_list

def add_layer_kernel_parsers(kernel_subparser):

    kernel_parser_list = []

    dotproduct = kernel_subparser.add_parser('dotproduct', description='''
        Use the dot product between vectors as your kernel function. NOTE: By using this 
        kernel you are not actually using the 'kernel trick' since we are explicitly 
        calculating the feature space''')
    dotproduct_subparsers = dotproduct.add_subparsers(dest='sub_metric', description='''
        Need to make a vectorized version of the protein conformation in order to use a 
        kernel function.''')
    dotproduct.metric_parser_list = add_basic_metric_parsers(dotproduct_subparsers)

    kernel_parser_list.extend(dotproduct.metric_parser_list)

    polynomial = kernel_subparser.add_parser('polynomial', description='''
        Use the polynomial kernel which is the dot product between vectors raised
        to some power, p.''')
    add_argument(polynomial, '-p', dest='poly_deg', type=float, help='Power to rais '
                 'dot product of vectors to.')
    polynomial_subparsers = polynomial.add_subparsers(dest='sub_metric', description='''
        Need to define a vectorized metric to use in computing dot products.''')
    polynomial.metric_parser_list = add_basic_metric_parsers(polynomial_subparsers)

    gaussian = kernel_subparser.add_parser('gaussian', description='''
        Use the gaussian kernel between vectors.''')
    add_argument(gaussian, '-s', dest='std_dev', default=1., type=float,
                 help='Standard deviation in the kernel function')
    gaussian_subparsers = gaussian.add_subparsers(dest='sub_metric', description='''
        Need to use a norm within the gaussian kernel function.''')
    gaussian.metric_parser_list = add_basic_metric_parsers(gaussian_subparsers)

    kernel_parser_list.extend(gaussian.metric_parser_list)

    return kernel_parser_list

def construct_layer_kernel(kernel_name, args):
    
    if kernel_name == 'dotproduct':
        sub_metric = construct_basic_metric(args.sub_metric, args)
    
        return DotProduct(sub_metric)

    elif kernel_name == 'gaussian':
        sub_metric = construct_basic_metric(args.sub_metric, args)

        return Gaussian(sub_metric, std_dev=args.std_dev)

    elif kernel_name == 'polynomial':
        sub_metric = construct_basic_metric(args.sub_metric, args)

        return Polynomial(sub_metric, power=args.poly_deg)

def construct_kernel(args):
    return construct_layer_kernel(args.kernel, args)

