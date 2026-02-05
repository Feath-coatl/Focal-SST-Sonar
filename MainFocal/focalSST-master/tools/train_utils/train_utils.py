import glob
import os

import torch
import tqdm
import time
import contextlib
import numpy as np

from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils

try:
    import torch.cuda.amp
except:
    # Make sure the torch version is latest enough to support mixed precision training
    pass

def check_batch_data(batch, logger, cur_it):
    return True


def check_loss_and_grad(loss, tb_dict, model, logger, cur_it):
    """检查loss和梯度中是否有NaN或Inf"""
    if torch.isnan(loss):
        logger.error(f"[DEBUG] NaN loss at iteration {cur_it}")
        logger.error(f"[DEBUG] tb_dict: {tb_dict}")
        return False
    if torch.isinf(loss):
        logger.error(f"[DEBUG] Inf loss at iteration {cur_it}")
        logger.error(f"[DEBUG] tb_dict: {tb_dict}")
        return False
    
    # 检查loss字典中的每个值
    for key, val in tb_dict.items():
        if isinstance(val, torch.Tensor):
            if torch.isnan(val).any():
                logger.error(f"[DEBUG] NaN in tb_dict['{key}'] at iteration {cur_it}")
                return False
            if torch.isinf(val).any():
                logger.error(f"[DEBUG] Inf in tb_dict['{key}'] at iteration {cur_it}")
                return False
        elif isinstance(val, (int, float)):
            if np.isnan(val):
                logger.error(f"[DEBUG] NaN in tb_dict['{key}'] at iteration {cur_it}")
                return False
            if np.isinf(val):
                logger.error(f"[DEBUG] Inf in tb_dict['{key}'] at iteration {cur_it}")
                return False
    return True


def safe_get_batch(dataloader_iter, train_loader, logger, cur_it, ckpt_save_dir, rank, max_retries=10):
    """安全地获取batch,捕获所有可能的异常并跳过有问题的数据"""
    retry_count = 0
    skipped_samples = []
    
    while retry_count < max_retries:
        try:
            # if retry_count == 0:
            #     logger.info(f"[DEBUG] Attempting to load batch {cur_it}...")
            # else:
            #     logger.warning(f"[DEBUG] Retry {retry_count}/{max_retries} for batch {cur_it}...")
            
            batch = next(dataloader_iter)
            
            # if skipped_samples:
            #     logger.warning(f"[DEBUG] Skipped {len(skipped_samples)} problematic samples before succeeding")
            
            # logger.info(f"[DEBUG] Successfully loaded batch {cur_it}")
            return batch, dataloader_iter
            
        except StopIteration:
            # logger.info(f"[DEBUG] StopIteration at {cur_it}, reinitializing dataloader...")
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            # logger.info(f"[DEBUG] Successfully reinitialized and loaded batch")
            return batch, dataloader_iter
            
        except (FloatingPointError, ZeroDivisionError, ValueError) as e:
            error_type = type(e).__name__
            # logger.error(f"[DEBUG] {error_type} in dataloader, retry {retry_count}: {e}")
            
            retry_count += 1
            skipped_samples.append(f"iter_{cur_it}_retry_{retry_count}")
            
            # 保存错误信息到日志文件
            if rank == 0:
                error_log_path = ckpt_save_dir / 'skipped_samples.log'
                with open(error_log_path, 'a') as f:
                    import datetime
                    f.write(f"[{datetime.datetime.now()}] Iteration {cur_it}, Retry {retry_count}: {error_type} - {str(e)[:200]}\n")
            
            # logger.warning(f"[DEBUG] Skipping problematic sample, attempting next one...")
            continue
            
        except RuntimeError as e:
            error_msg = str(e)
            logger.error(f"[DEBUG] RuntimeError in dataloader: {e}")
            
            if "worker" in error_msg.lower():
                logger.error(f"[DEBUG] Error in worker process")
                retry_count += 1
                if retry_count < max_retries:
                    continue
            raise
            
        except Exception as e:
            logger.error(f"[DEBUG] Unexpected error: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
            
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"[DEBUG] Retrying... ({retry_count}/{max_retries})")
                continue
            else:
                raise
    
    logger.error(f"[DEBUG] Failed to load batch after {max_retries} retries")
    raise RuntimeError(f"Failed to load valid batch after {max_retries} retries at iteration {cur_it}")



def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,
                    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None,
                    total_epochs=None, ckpt_save_dir=None, ckpt_save_time_interval=300, show_gpu_stat=False, fp16=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        loss_disp = common_utils.AverageMeter()
        # just for centerhead
        hm_loss_disp = common_utils.AverageMeter()
        loc_loss_disp = common_utils.AverageMeter()
        rcnn_cls_loss_disp = common_utils.AverageMeter()
        rcnn_reg_loss_disp = common_utils.AverageMeter()


    amp_ctx = contextlib.nullcontext()
    if fp16:
        scaler = torch.cuda.amp.grad_scaler.GradScaler(init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0**16))
        amp_ctx = torch.cuda.amp.autocast()


    # end = time.time()
    # for cur_it in range(start_it, total_it_each_epoch):
    #     try:
    #         batch = next(dataloader_iter)

    #         # DEBUG: 检查batch数据
    #         if logger is not None:
    #             if not check_batch_data(batch, logger, cur_it):
    #                 logger.error(f"[DEBUG] Problematic batch at iteration {cur_it}, saving debug info...")
    #                 if rank == 0:
    #                     # 保存问题数据用于离线调试
    #                     debug_path = ckpt_save_dir / f'debug_batch_iter_{cur_it}.pth'
    #                     torch.save(batch, debug_path)
    #                     logger.error(f"[DEBUG] Saved problematic batch to {debug_path}")
    #                 raise RuntimeError(f"Invalid data in batch at iteration {cur_it}")

    #     except StopIteration:
    #         dataloader_iter = iter(train_loader)
    #         batch = next(dataloader_iter)
    #         print('new iters')

    end = time.time()
    for cur_it in range(start_it, total_it_each_epoch):
        # DEBUG: 使用安全的batch加载函数
        # if logger is not None:
            # logger.info(f"[DEBUG] Starting iteration {cur_it}/{total_it_each_epoch}")
        
        # 使用包装函数安全地获取batch
        if logger is not None:
            batch, dataloader_iter = safe_get_batch(dataloader_iter, train_loader, logger, cur_it, ckpt_save_dir, rank)
        else:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_loader)
                batch = next(dataloader_iter)
                print('new iters')
        
        # DEBUG: 检查batch数据
        if logger is not None:
            # logger.info(f"[DEBUG] Checking batch data at iteration {cur_it}...")
            if not check_batch_data(batch, logger, cur_it):
                logger.error(f"[DEBUG] Problematic batch at iteration {cur_it}, saving debug info...")
                if rank == 0:
                    # 保存问题数据用于离线调试
                    debug_path = ckpt_save_dir / f'debug_batch_iter_{cur_it}.pth'
                    torch.save(batch, debug_path)
                    logger.error(f"[DEBUG] Saved problematic batch to {debug_path}")
                raise RuntimeError(f"Invalid data in batch at iteration {cur_it}")
            # logger.info(f"[DEBUG] Batch data check passed for iteration {cur_it}")

        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        
        # 梯度累积配置
        grad_accumulation_steps = getattr(optim_cfg, 'GRAD_ACCUMULATION_STEPS', 1)
        is_accumulation_step = (cur_it + 1) % grad_accumulation_steps != 0
        
        # 只在第一步清零梯度
        if cur_it % grad_accumulation_steps == 0:
            optimizer.zero_grad()

        with amp_ctx:
            loss, tb_dict, disp_dict = model_func(model, batch)
            
            # 梯度累积：损失除以累积步数
            if grad_accumulation_steps > 1:
                loss = loss / grad_accumulation_steps

            # DEBUG: 检查loss和tb_dict
            if logger is not None:
                if not check_loss_and_grad(loss, tb_dict, model, logger, cur_it):
                    logger.error(f"[DEBUG] Problematic loss at iteration {cur_it}")
                    if rank == 0:
                        debug_path = ckpt_save_dir / f'debug_loss_iter_{cur_it}.pth'
                        torch.save({'loss': loss.item(), 'tb_dict': tb_dict, 'batch': batch}, debug_path)
                        logger.error(f"[DEBUG] Saved debug info to {debug_path}")
                    raise RuntimeError(f"Invalid loss at iteration {cur_it}")

            if fp16:
                assert loss.dtype is torch.float32
                scaler.scale(loss).backward()
                
                # 只在累积完成后更新参数
                if not is_accumulation_step:
                    scaler.unscale_(optimizer)
                    total_norm = clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_norm = 0.0
            else:
                loss.backward()
                
                # 只在累积完成后更新参数
                if not is_accumulation_step:
                    total_norm = clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                    optimizer.step()
                else:
                    total_norm = 0.0

        # 只在真正更新参数后增加accumulated_iter
        if not is_accumulation_step:
            accumulated_iter += 1
        # assert not torch.isnan(loss)

        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            loss_disp.update(loss.item())
            
            # for centerhead
            if 'hm_loss_head_0' in list(tb_dict.keys()) and 'loc_loss_head_0' in list(tb_dict.keys()):
                hm_loss_disp.update(tb_dict['hm_loss_head_0'])
                loc_loss_disp.update(tb_dict['loc_loss_head_0'])
                disp_dict.update({
                'loss_hm': f'{hm_loss_disp.avg:.4f}', 'loss_loc': f'{loc_loss_disp.avg:.4f}'})
            if 'rcnn_loss_reg' in list(tb_dict.keys()) and 'rcnn_loss_cls' in list(tb_dict.keys()):
                rcnn_cls_loss_disp.update(tb_dict['rcnn_loss_cls'])
                rcnn_reg_loss_disp.update(tb_dict['rcnn_loss_reg'])
                disp_dict.update({
                'loss_rcnn_cls': f'{rcnn_cls_loss_disp.avg:.4f}', 'loss_rcnn_reg': f'{rcnn_reg_loss_disp.avg:.4f}'})
            disp_dict.update({
                'loss': loss_disp.avg, 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})',
                'norm': total_norm.item() if hasattr(total_norm, 'item') else total_norm
            })

            if use_logger_to_record:
                if (accumulated_iter % logger_iter_interval == 0 and cur_it != start_it) or cur_it + 1 == total_it_each_epoch:
                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)

                    disp_str = ', '.join([f'{key}={val}' for key, val in disp_dict.items() if key != 'lr'])
                    disp_str += f', lr={disp_dict["lr"]}'
                    batch_size = batch.get('batch_size', None)
                    logger.info(f'epoch: {cur_epoch}/{total_epochs}, acc_iter={accumulated_iter}, cur_iter={cur_it}/{total_it_each_epoch}, batch_size={batch_size}, '
                                f'time_cost(epoch): {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)}, '
                                f'time_cost(all): {tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}, '
                                f'{disp_str}')
                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                        # To show the GPU utilization, please install gpustat through "pip install gpustat"
                        gpu_info = os.popen('gpustat').read()
                        logger.info(gpu_info)
                    
                    loss_disp.reset()  # WHY
                    hm_loss_disp.reset()
                    loc_loss_disp.reset()
                    rcnn_cls_loss_disp.reset()
                    rcnn_reg_loss_disp.reset()
            else:
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                # tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)

            # save intermediate ckpt every {ckpt_save_time_interval} seconds
            time_past_this_epoch = pbar.format_dict['elapsed']
            if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                ckpt_name = ckpt_save_dir / 'latest_model'
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                )
                logger.info(f'Save latest model to {ckpt_name}')
                ckpt_save_cnt += 1

    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False,
                use_logger_to_record=False, logger=None, logger_iter_interval=None, ckpt_save_time_interval=None, show_gpu_stat=False, fp16=False, cfg=None):
    accumulated_iter = start_iter

    # DEBUG: 检查DataLoader配置
    if logger is not None:
        logger.info(f"[DEBUG] DataLoader info:")
        logger.info(f"[DEBUG]   - num_workers: {train_loader.num_workers}")
        logger.info(f"[DEBUG]   - batch_size: {train_loader.batch_size}")
        logger.info(f"[DEBUG]   - dataset size: {len(train_loader.dataset)}")
        logger.info(f"[DEBUG]   - pin_memory: {train_loader.pin_memory}")
        if train_loader.num_workers > 0:
            logger.warning(f"[DEBUG] Using multi-process data loading (num_workers={train_loader.num_workers})")
            logger.warning(f"[DEBUG] If you encounter worker errors, try setting num_workers=0 for easier debugging")

    augment_disable_flag = False
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            
            hook_config = cfg.get('HOOK', None) 
            if hook_config is not None:
                DisableAugmentationHook = hook_config.get('DisableAugmentationHook', None)
                if DisableAugmentationHook is not None:
                    num_last_epochs = cfg.HOOK.DisableAugmentationHook.NUM_LAST_EPOCHS
                    if (total_epochs - num_last_epochs) <= cur_epoch and not augment_disable_flag:
                        from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
                        from pathlib import Path
                        DISABLE_AUG_LIST = cfg.HOOK.DisableAugmentationHook.DISABLE_AUG_LIST
                        dataset_cfg=cfg.DATA_CONFIG
                        # This hook turns off some data augmentation strategies. 
                        logger.info(f'Disable augmentations: {DISABLE_AUG_LIST}')
                        dataset_cfg.DATA_AUGMENTOR.DISABLE_AUG_LIST = DISABLE_AUG_LIST
                        class_names=cfg.CLASS_NAMES
                        root_path = Path(dataset_cfg.DATA_PATH)
                        new_data_augmentor = DataAugmentor(root_path, dataset_cfg.DATA_AUGMENTOR, class_names, logger=logger)
                        dataloader_iter._dataset.data_augmentor = new_data_augmentor
                        augment_disable_flag = True


            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,

                cur_epoch=cur_epoch, total_epochs=total_epochs,
                use_logger_to_record=use_logger_to_record,
                logger=logger, logger_iter_interval=logger_iter_interval,
                ckpt_save_dir=ckpt_save_dir, ckpt_save_time_interval=ckpt_save_time_interval,
                show_gpu_stat=show_gpu_stat,
                fp16=fp16
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)
