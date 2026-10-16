        for _p in extra_pretrained_params:
            if _p in n:
                not_pretrained_params.remove(n)

    return not_pretrained_params


def get_param_id_dict(model: VisionTransformer, patterns: list[str]) -> dict[int, dict[Literal['name', 'shape'], str | list[int]]]:
    param_id_dict = {}
    for n, p in model.named_parameters():
        if p.requires_grad and any([_s in n for _s in patterns]):
            param_id_dict[id(p)] = {'name': n, 'shape': list(p.shape)}
    
    assert len(param_id_dict) > 0, f"{param_id_dict}"
    return param_id_dict


def get_head_dim_arg_dict(GVM: GlobalVarsManager, args: argparse.Namespace) -> dict[Literal['num_classes'], int]:
    head_dim_arg_dict = {}
    head_dim_type = args.head_dim_type

    match args.logit_type:
        case 'sim_imgtext':
            assert head_dim_type in ('pretrained', 'text_dim')
        case 'head_out':
            assert head_dim_type in ('task_classes')

    match head_dim_type:
        case 'task_classes':
            head_dim_arg_dict['num_classes'] = len(current_task_classes) if args.seperate_head else len(GVM.cl_mngr.sofar_task_classes)
        case 'pretrained':
            pass
        case 'text_dim':
            head_dim_arg_dict['num_classes'] = 512
        case _:
            raise ValueError(head_dim_type)
    return head_dim_arg_dict


def modify_head(GVM: GlobalVarsManager, model: VisionTransformer, training: bool, **kwargs):
    args: argparse.Namespace = GVM.args

    if training:
        _target_classes = kwargs['task_classes'] if args.seperate_head else GVM.cl_mngr.sofar_task_classes
    else:
        _target_classes = GVM.cl_mngr.sofar_task_classes


    if model.head.out_features != len(_target_classes):
        _mh = deepcopy(model.head)
        _mdevice = _mh.weight.device
        _mdtype = _mh.weight.dtype
        model.head = _mh.__class__(_mh.in_features, len(_target_classes), _mh.bias is not None, _mdevice, _mdtype)
        model.head.requires_grad_(_mh.weight.requires_grad)

        if training:
            assert model.head.weight.requires_grad
        else:
            assert _mh.out_features == len(GVM.cl_mngr.current_task_classes), f"{_mh.out_features}, {len(GVM.cl_mngr.current_task_classes)}"
            _hw = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.weight'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
            assert model.head.weight.data.shape == _hw.shape
            model.head.weight.data = _hw

            if _mh.bias is not None:
                _hb = torch.cat([GVM.param_dict[f'task_params_{_t}']['head.bias'].data.to(_mdevice, _mdtype) for _t in range(GVM.cl_mngr.current_taskid + 1)])
                assert model.head.bias.data.shape == _hb.shape
                model.head.bias.data = _hb

    return model






if __name__ == "__main__":
    args = get_args()
    seed_etc_options(args.seed)

    GVM = GlobalVarsManager()
    GVM.init_from_args(args)

    GVM.cache_dict['exp_start_time'] = ttime()
    lr, rank, alpha = args.lr, args.rank, args.alpha
    wandb_name = f"{args.seed}"
    wandb.init(
        name=wandb_name,
        mode="offline",
        group="test",
        project=f"{args.dataset}_{args.num_tasks}",
    )
    for taskid, current_task_classes in GVM.cl_mngr:
        print(f"{'#'*90} Task: [{taskid+1}/{GVM.cl_mngr.num_tasks}] {'#'*90}")
        print(f"Current classes ({len(current_task_classes)}): {current_task_classes}")

        if  taskid == 0:
            _other_args_dict = misc.get_specific_args_dict(args, 'logit_')
            _head_dim_arg_dict = get_head_dim_arg_dict(GVM, args)

            model: VisionTransformer = timm.create_model(args.model, pretrained=True, pretrained_strict=False, **_head_dim_arg_dict,
                                                         other_args_dict=_other_args_dict)
            GVM.cache_dict['pretrained_cfg'] = deepcopy(model.pretrained_cfg)
            apply_lora(Namespace(encoder="both",params=["k", "v"],r=args.rank,alpha=args.alpha,task_id=0,num_tasks=args.num_tasks,),model,)
        model.num_tasks = args.num_tasks
        param = ['k','v']

        r=  args.rank
        alpha = args.alpha
        args.training_string = ['lora','head']
        if args.consecutive_training and taskid > 0:
            pass
        GVM._label_maps(taskid, current_task_classes)
        GVM.cache_dict['training_string'] = args.training_string
        model = train_one_task(GVM, taskid, current_task_classes, model,args)
        _compute_class_mean(GVM, model, current_task_classes) # lưu prototype cuối mỗi task
        GVM.old_model = deepcopy(model).eval() 
        GVM.old_model.requires_grad_(False)
        evaluate_tasks_sofar(GVM, taskid, model)
        task_ending_info(GVM)
